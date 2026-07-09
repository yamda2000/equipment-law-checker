"""ケースメモリ（Case-Based Reasoning / 事例ベース推論）

担当者がレポートを承認した案件を「事例」としてベクトル保存し、
新しい案件の情報整理・分析フェーズで類似事例を想起して LLM に渡す。
承認済みの結論だけを蓄積するため、誤った推測が伝播しにくい。

- 保存: report_node の承認時に save_case()（設備プロファイルを埋め込み）
- 想起: analysis_node で find_similar_cases()（新案件のプロファイルで類似検索）
- 事例の全文は cases.json に保持し、ベクトルストアは検索用プロファイルのみ持つ
- ベクトルストア・埋め込みは社内文書検索と同じ基盤（internal_docs）を共有
"""

import json
import logging
import datetime
import threading

from langchain_core.documents import Document

import backend.tools.internal_docs as _idocs

logger = logging.getLogger(__name__)

# 検索用プロファイルの項目名（workflow.HEARING_FIELD_JA と対応。
# workflow を import すると循環参照になるためここに定義する）
_FIELD_JA = {
    "equipment_type":     "設備の種類",
    "installation_place": "設置場所",
    "operation_purpose":  "用途・目的",
    "scheduled_date":     "稼働開始予定日",
    "chemicals":          "薬品・溶剤・ガス・燃料",
    "fire_exhaust":       "火気・熱源・排気・粉じん",
    "wastewater":         "排水・廃液・廃棄物",
    "noise_vibration":    "騒音・振動",
    "radiation":          "放射線・X線",
    "construction":       "建屋改修・電気工事・配管工事",
    "additional_info":    "その他の情報",
}

_LOCK = threading.Lock()
_STORE = None


def _cases_path():
    return _idocs.INDEX_DIR / "cases.json"


def _faiss_dir():
    return _idocs.INDEX_DIR / "case_faiss"


def _load_store():
    global _STORE
    if _STORE is not None:
        return _STORE
    if _idocs._backend() == "faiss":
        from langchain_community.vectorstores import FAISS
        if (_faiss_dir() / "index.faiss").exists():
            _STORE = FAISS.load_local(
                str(_faiss_dir()), _idocs._embeddings(),
                allow_dangerous_deserialization=True,
            )
    else:
        from langchain_community.vectorstores import Chroma
        _STORE = Chroma(
            collection_name="case_memory",
            embedding_function=_idocs._embeddings(),
            persist_directory=str(_idocs.INDEX_DIR / "chroma"),
        )
    return _STORE


def _persist_if_faiss() -> None:
    if _idocs._backend() == "faiss" and _STORE is not None:
        _faiss_dir().mkdir(parents=True, exist_ok=True)
        _STORE.save_local(str(_faiss_dir()))


def _load_cases() -> list[dict]:
    try:
        return json.loads(_cases_path().read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_cases(cases: list[dict]) -> None:
    _idocs.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _cases_path().write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _profile_text(equipment_info: dict) -> str:
    """設備情報を類似検索用のプロファイル文字列にする。"""
    lines = []
    for key, ja in _FIELD_JA.items():
        val = str(equipment_info.get(key, "") or "").strip()
        if val:
            lines.append(f"{ja}: {val}")
    return "\n".join(lines)


# ─── 保存（Retain）───────────────────────────────────────────────
def save_case(
    case_id: str,
    equipment_info: dict,
    law_items: list,
    excluded_laws: list | None = None,
    summary: str = "",
) -> bool:
    """承認済み案件を事例として保存する。同一 case_id は上書き
    （再調査→再承認で結論が変わった場合に最新を残す）。"""
    if not case_id or not equipment_info:
        return False
    profile = _profile_text(equipment_info)
    if not profile:
        return False

    record = {
        "case_id":  case_id,
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d"),
        "profile":  profile,
        "summary":  summary,
        "laws": [
            {
                "law_name":         l.get("law_name", ""),
                "priority":         l.get("priority", ""),
                "applicability":    str(l.get("applicability", ""))[:200],
                "deliveries":       [d.get("item", "") for d in l.get("deliveries", []) if d.get("item")],
                "internal_actions": [a.get("item", "") for a in l.get("internal_actions", []) if a.get("item")],
            }
            for l in (law_items or [])
        ],
        "excluded_laws": [
            {"law_name": e.get("law_name", ""), "reason": str(e.get("reason", ""))[:200]}
            for e in (excluded_laws or [])
        ],
    }

    global _STORE
    with _LOCK:
        cases = _load_cases()
        store = _load_store()
        if any(c.get("case_id") == case_id for c in cases) and store is not None:
            try:
                store.delete(ids=[case_id])
            except Exception:
                logger.exception("ケースメモリの旧事例削除に失敗: %s", case_id)

        doc = Document(page_content=profile, metadata={"case_id": case_id})
        if store is None:
            # faiss は空インデックスを作れないため、初回保存時に生成する
            from langchain_community.vectorstores import FAISS
            _STORE = FAISS.from_documents([doc], _idocs._embeddings(), ids=[case_id])
        else:
            store.add_documents([doc], ids=[case_id])
        _persist_if_faiss()

        _save_cases([c for c in cases if c.get("case_id") != case_id] + [record])
    return True


# ─── 想起（Retrieve）─────────────────────────────────────────────
def find_similar_cases(
    equipment_info: dict, k: int = 2, exclude_case_id: str = "",
) -> list[dict]:
    """新案件の設備情報に類似する過去事例を返す（類似度順・最大k件）。"""
    cases = _load_cases()
    if not cases:
        return []
    by_id = {c["case_id"]: c for c in cases}
    query = _profile_text(equipment_info)
    if not query:
        return []
    try:
        store = _load_store()
        if store is None:
            return []
        hits = store.similarity_search(query, k=min(k + 1, len(cases) + 1))
    except Exception:
        logger.exception("ケースメモリの類似検索に失敗")
        return []

    found: list[dict] = []
    for d in hits:
        cid = d.metadata.get("case_id", "")
        if cid and cid != exclude_case_id and cid in by_id:
            found.append(by_id[cid])
        if len(found) >= k:
            break
    return found


_PRIORITY_JA = {"required": "必須対応", "check": "要確認", "pending": "確認中"}


def format_cases_for_prompt(cases: list[dict]) -> str:
    """類似事例を LLM プロンプト注入用のテキストに整形する。"""
    if not cases:
        return ""
    blocks = []
    for i, c in enumerate(cases, 1):
        lines = [f"### 類似案件{i}（{c.get('case_id', '?')}・{c.get('saved_at', '')} 承認済み）"]
        lines.append(f"【設備プロファイル】\n{c.get('profile', '')}")
        laws = c.get("laws", [])
        if laws:
            lines.append("【該当した法令と対応】")
            for l in laws:
                pr = _PRIORITY_JA.get(l.get("priority", ""), l.get("priority", ""))
                s = f"- {l.get('law_name', '')}（{pr}）: {l.get('applicability', '')}"
                if l.get("deliveries"):
                    s += f"\n  届出: {'、'.join(l['deliveries'])}"
                if l.get("internal_actions"):
                    s += f"\n  社内対応: {'、'.join(l['internal_actions'])}"
                lines.append(s)
        excluded = c.get("excluded_laws", [])
        if excluded:
            lines.append("【非該当と判断した法令】")
            for e in excluded:
                lines.append(f"- {e.get('law_name', '')}: {e.get('reason', '')}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ─── 管理 ─────────────────────────────────────────────────────────
def list_cases() -> list[dict]:
    """保存済み事例の一覧（新しい順）。"""
    cases = _load_cases()
    out = []
    for c in reversed(cases):
        eq_type = ""
        for line in c.get("profile", "").splitlines():
            if line.startswith("設備の種類:"):
                eq_type = line.split(":", 1)[1].strip()
                break
        out.append({
            "case_id":  c.get("case_id", ""),
            "saved_at": c.get("saved_at", ""),
            "equipment_type": eq_type,
            "law_count": len(c.get("laws", [])),
        })
    return out


def delete_case(case_id: str) -> bool:
    """事例をケースメモリから削除する。"""
    with _LOCK:
        cases = _load_cases()
        if not any(c.get("case_id") == case_id for c in cases):
            return False
        store = _load_store()
        if store is not None:
            try:
                store.delete(ids=[case_id])
                _persist_if_faiss()
            except Exception:
                logger.exception("ケースメモリからの削除に失敗: %s", case_id)
        _save_cases([c for c in cases if c.get("case_id") != case_id])
    return True
