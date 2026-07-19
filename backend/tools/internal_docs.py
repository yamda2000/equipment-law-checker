"""社内文書のベクトル検索（Agentic RAG 用）

事前に登録した社内文書（社内規定・基準・過去の届出事例・手続きマニュアル等）を
チャンク分割・埋め込みして永続化し、調査フェーズの Agentic 検索ループから
e-Gov API・Web検索と並ぶ検索タイプ（search_type="internal"）として利用する。

- ベクトルストア: VECTOR_STORE 環境変数で切替（chroma=既定 / faiss）
- 埋め込み: LLM_MODE 環境変数で切替（poc=OpenAI / prod=Azure OpenAI）
- 登録済みファイルの一覧・重複判定・削除のため registry.json を併置する
"""

import os
import json
import hashlib
import datetime
import logging
import threading
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# インデックスの永続化先（プロジェクト直下）
INDEX_DIR = Path(__file__).resolve().parents[2] / "internal_docs_index"
REGISTRY_PATH = INDEX_DIR / "registry.json"

# 1ファイルあたりの登録テキスト上限（埋め込みコストの暴走防止。超過分は切り捨て）
MAX_DOC_CHARS = 50000

# チャンク分割設定（日本語文書向けの区切り優先順）
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", "。", "、", " ", ""],
)

_LOCK = threading.Lock()
_STORE = None  # プロセス内キャッシュ


# ─── 埋め込みファクトリ（LLM_MODE で OpenAI / Azure を切替）──────
def _embeddings():
    """EMBEDDING 系の環境変数を優先し、未設定なら LLM 系の設定にフォールバックする
    （.env.example の定義と一致させる。埋め込みが LLM と別リソースの構成に対応）。
    ※ 埋め込みモデルを変更した場合、既存インデックスと次元が合わなくなるため
    　 社内文書・ケースメモリの再登録が必要。"""
    mode = os.getenv("LLM_MODE", "poc").lower()
    if mode == "prod":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("PROD_EMBEDDING_DEPLOYMENT", ""),
            azure_endpoint=(
                os.getenv("PROD_EMBEDDING_ENDPOINT", "")
                or os.getenv("PROD_LLM_ENDPOINT", "")
            ),
            api_key=(
                os.getenv("PROD_EMBEDDING_API_KEY", "")
                or os.getenv("PROD_LLM_API_KEY", "")
            ),
            openai_api_version=(
                os.getenv("PROD_EMBEDDING_API_VERSION", "")
                or os.getenv("PROD_LLM_API_VERSION", "2024-02-01")
            ),
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=os.getenv("POC_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=(
            os.getenv("POC_EMBEDDING_API_KEY")
            or os.getenv("POC_LLM_API_KEY")
        ),
    )


def _backend() -> str:
    return os.getenv("VECTOR_STORE", "chroma").lower()


# ─── ベクトルストア ───────────────────────────────────────────────
def _faiss_dir() -> Path:
    return INDEX_DIR / "faiss"


def _load_store():
    """既存インデックスをロードする。未作成なら None（faiss）または空ストア（chroma）。"""
    global _STORE
    if _STORE is not None:
        return _STORE
    if _backend() == "faiss":
        from langchain_community.vectorstores import FAISS
        if (_faiss_dir() / "index.faiss").exists():
            # 自前で保存したローカルインデックスのため deserialization を許可する
            _STORE = FAISS.load_local(
                str(_faiss_dir()), _embeddings(),
                allow_dangerous_deserialization=True,
            )
    else:
        from langchain_chroma import Chroma
        _STORE = Chroma(
            collection_name="internal_docs",
            embedding_function=_embeddings(),
            persist_directory=str(INDEX_DIR / "chroma"),
        )
    return _STORE


def _persist_if_faiss() -> None:
    if _backend() == "faiss" and _STORE is not None:
        _faiss_dir().mkdir(parents=True, exist_ok=True)
        _STORE.save_local(str(_faiss_dir()))


# ─── 登録ファイル台帳 ─────────────────────────────────────────────
def _load_registry() -> list[dict]:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_registry(reg: list[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_registered() -> list[dict]:
    """登録済みファイルの一覧（name / chunks / registered_at）。"""
    return [
        {"name": r["name"], "chunks": r.get("chunks", 0),
         "registered_at": r.get("registered_at", "")}
        for r in _load_registry()
    ]


def internal_docs_available() -> bool:
    """社内文書が1件でも登録されているか（検索タイプの有効判定に使う）。"""
    return bool(_load_registry())


# ─── 登録・削除 ───────────────────────────────────────────────────
def ingest_files(files: list[tuple[str, bytes]]) -> dict:
    """[(ファイル名, バイト列), ...] を抽出→分割→埋め込みして登録する。
    返り値: {"added": [名前], "skipped": [名前(登録済み)],
             "failed": [名前(抽出不可)], "truncated": [名前(上限で切り捨て)]}
    """
    # 循環importを避けるため遅延import（doc_intake → workflow → tools）
    from backend.doc_intake import extract_text_from_file

    global _STORE
    added, skipped, failed, truncated = [], [], [], []

    with _LOCK:
        reg = _load_registry()
        known_hashes = {r["sha1"] for r in reg}

        for name, data in files:
            sha1 = hashlib.sha1(data).hexdigest()
            if sha1 in known_hashes:
                skipped.append(name)
                continue
            text = extract_text_from_file(name, data)
            if not text.strip():
                failed.append(name)
                continue
            if len(text) > MAX_DOC_CHARS:
                text = text[:MAX_DOC_CHARS]
                truncated.append(name)

            chunks = _SPLITTER.split_text(text)
            ids = [f"{sha1}-{i}" for i in range(len(chunks))]
            docs = [
                Document(
                    page_content=c,
                    metadata={"source": name, "chunk_no": i + 1},
                )
                for i, c in enumerate(chunks)
            ]

            store = _load_store()
            if store is None:
                # faiss は空インデックスを作れないため、初回登録時に生成する
                from langchain_community.vectorstores import FAISS
                _STORE = FAISS.from_documents(docs, _embeddings(), ids=ids)
            else:
                store.add_documents(docs, ids=ids)
            _persist_if_faiss()

            reg.append({
                "name": name,
                "sha1": sha1,
                "chunk_ids": ids,
                "chunks": len(chunks),
                "registered_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            known_hashes.add(sha1)
            added.append(name)

        if added:
            _save_registry(reg)

    return {"added": added, "skipped": skipped, "failed": failed, "truncated": truncated}


def delete_file(name: str) -> bool:
    """登録済みファイルをインデックスと台帳から削除する。"""
    global _STORE
    with _LOCK:
        reg = _load_registry()
        entry = next((r for r in reg if r["name"] == name), None)
        if entry is None:
            return False
        store = _load_store()
        if store is not None and entry.get("chunk_ids"):
            try:
                store.delete(ids=entry["chunk_ids"])
                _persist_if_faiss()
            except Exception:
                logger.exception("社内文書インデックスからの削除に失敗: %s", name)
        _save_registry([r for r in reg if r["name"] != name])
    return True


def delete_all() -> int:
    """登録済みの全ファイルをインデックスと台帳から削除する。返り値は削除ファイル数。"""
    global _STORE
    with _LOCK:
        reg = _load_registry()
        if not reg:
            return 0
        store = _load_store()
        all_ids = [i for r in reg for i in r.get("chunk_ids", [])]
        if store is not None and all_ids:
            try:
                store.delete(ids=all_ids)
                _persist_if_faiss()
            except Exception:
                logger.exception("社内文書インデックスの全削除に失敗")
        _save_registry([])
    return len(reg)


# ─── 検索 ─────────────────────────────────────────────────────────
def retrieve_chunks(query: str, k: int = 4) -> list[dict]:
    """Agentic RAG サブエージェント用の低レベル検索。
    チャンク全文とID付きで返す（関連性評価に全文が必要なため）。
    未登録・失敗時は空リスト。
    """
    if not internal_docs_available():
        return []
    try:
        store = _load_store()
        if store is None:
            return []
        hits = store.similarity_search(query, k=k)
        return [
            {
                "id": f"{d.metadata.get('source', '?')}#{d.metadata.get('chunk_no', '?')}",
                "source_file": d.metadata.get("source", "?"),
                "chunk_no": d.metadata.get("chunk_no", 0),
                "content": d.page_content,
            }
            for d in hits
        ]
    except Exception:
        logger.exception("社内文書チャンク検索に失敗: %s", query)
        return []


def search_internal_docs(query: str, k: int = 5) -> list[dict]:
    """社内文書をベクトル検索し、search_web と同形式の結果リストを返す。
    未登録・インデックス無しは空リスト。失敗時は source="error" の1件を返す
    （呼び出し側の _process_web_results がエラー表示に変換する）。
    """
    if not internal_docs_available():
        return []
    try:
        store = _load_store()
        if store is None:
            return []
        hits = store.similarity_search(query, k=k)
        return [
            {
                "title": (
                    f"社内文書: {d.metadata.get('source', '?')}"
                    f"（抜粋{d.metadata.get('chunk_no', '?')}）"
                ),
                "snippet": d.page_content[:300],
                "url": "",
                "source": "社内文書",
            }
            for d in hits
        ]
    except Exception as e:
        logger.exception("社内文書検索に失敗: %s", query)
        return [{"title": f"社内文書検索エラー: {e}", "snippet": "", "url": "", "source": "error"}]
