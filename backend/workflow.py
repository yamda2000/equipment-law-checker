"""
LangGraph ワークフロー
フェーズ: hearing → analysis → search → synthesis → report → complete
Human in the loop: analysis / synthesis / report の3箇所で interrupt
LLM: PoC=OpenAI API / 本番=Azure OpenAI (LLM_MODE 環境変数で切り替え)
"""

import os
import json
import uuid
import datetime
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from pydantic import BaseModel, Field

try:
    from langgraph.config import get_stream_writer
except Exception:  # 古いバージョン互換
    get_stream_writer = None

from backend.state import AppState
from backend.prompts import HEARING_SYSTEM, ANALYSIS_SYSTEM, SYNTHESIS_SYSTEM, SEARCH_AGENT_SYSTEM
from backend.tools.egov import search_laws_by_keyword
from backend.tools.web_search import search_web
from backend.report_gen import generate_html_report


# ─── LLM ファクトリ ───────────────────────────────────────────────
def _llm(temperature: float = 0.0, max_tokens: int = 4096):
    """LLM_MODE に応じて OpenAI または Azure OpenAI を返す"""
    mode = os.getenv("LLM_MODE", "poc").lower()

    if mode == "prod":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=os.getenv("PROD_LLM_DEPLOYMENT", ""),
            azure_endpoint=os.getenv("PROD_LLM_ENDPOINT", ""),
            api_key=os.getenv("PROD_LLM_API_KEY", ""),
            openai_api_version=os.getenv("PROD_LLM_API_VERSION", "2024-02-01"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("POC_LLM_MODEL", "gpt-4o"),
            api_key=os.getenv("POC_LLM_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ─── 構造化出力スキーマ ───────────────────────────────────────────
class AnalysisResult(BaseModel):
    issues: list[str] = Field(description="確認が必要な論点リスト")
    unknown_items: list[str] = Field(description="不明・未定情報と影響リスト")
    search_keywords: list[str] = Field(description="e-Gov API 検索キーワードリスト（6〜10個）")
    search_plan: str = Field(description="調査方針（3〜5行）")
    analysis_summary: str = Field(description="設備情報の分析サマリー")


MAX_SEARCH_ITERATIONS = 20


class SearchAction(BaseModel):
    done: bool = Field(description="True=収集完了, False=追加検索が必要")
    search_type: Literal["egov", "web"] = Field(default="egov", description="egov: e-Gov法令API / web: Web検索")
    query: str = Field(default="", description="次の検索クエリ（done=Falseの場合は必須）")
    reason: str = Field(description="判断理由")


class DeliveryItem(BaseModel):
    item: str = Field(description="届出・申請の名称")
    authority: str = Field(description="届出先・申請先機関")
    deadline: str = Field(description="届出期限（例：設置前、稼働前）")
    priority: str = Field(description="required | check | pending")
    law_article: str = Field(default="", description="根拠条文（例：第10条第1項）。特定できる場合のみ")


class InternalActionItem(BaseModel):
    item: str = Field(description="社内対応事項の内容")
    responsible: str = Field(description="担当部署・担当者（例：安全環境部、設備担当者）")
    deadline: str = Field(description="対応期限の目安（例：稼働3ヶ月前）")


class LawItem(BaseModel):
    law_name: str = Field(description="法令名（例：消防法、労働安全衛生法）")
    applicability: str = Field(description="この設備にこの法令が適用される理由（1〜2行）")
    priority: str = Field(description="required | check | pending")
    relevant_articles: list[str] = Field(default_factory=list, description="関連条文番号リスト（例：['第10条', '第11条第1項']）。特定できるものだけ列挙")
    deliveries: list[DeliveryItem] = Field(default_factory=list, description="届出・申請事項")
    internal_actions: list[InternalActionItem] = Field(default_factory=list, description="社内対応事項")


class RiskCount(BaseModel):
    required: int = Field(description="必須対応の法令件数")
    check: int = Field(description="要確認の法令件数")
    pending: int = Field(description="確認中の法令件数")


class SynthesisResult(BaseModel):
    law_items: list[LawItem] = Field(description="法令別の対応事項リスト")
    summary: str = Field(description="調査結果の総括（3〜5行）")
    risk_count: RiskCount = Field(description="優先度別の法令件数")


class RefineResult(BaseModel):
    law_items: list[LawItem] = Field(description="文面修正後の法令別対応事項リスト")


# ─── 再開（resume）デシジョン解析ヘルパー ────────────────────────────
_REINVEST_PREFIXES = ("reinvestigate:", "reinvestigate：")
_REFINE_PREFIXES   = ("refine:", "refine：")


def _extract_after(decision, prefixes: tuple) -> str:
    """「prefix: 本文」形式の resume 値から本文を取り出す。該当なしは空文字。"""
    if isinstance(decision, str):
        for p in prefixes:
            if decision.startswith(p):
                return decision.split(p, 1)[1].strip()
    return ""


def _is_reinvestigate(decision) -> bool:
    return isinstance(decision, str) and any(
        decision.startswith(p) for p in _REINVEST_PREFIXES
    )


def _merge_note(existing: str, extra: str) -> str:
    extra = (extra or "").strip()
    existing = existing or ""
    if not extra:
        return existing
    return f"{existing}\n{extra}".strip() if existing else extra


# ─── ヒアリングツール定義（OpenAI/Azure 形式）────────────────────
COMPLETE_HEARING_TOOL = {
    "type": "function",
    "function": {
        "name": "complete_hearing",
        "description": "設備情報の収集が十分に完了したと判断した場合に呼び出す。全項目について回答が揃ったとき（不明・未定も含む）に使用する。",
        "parameters": {
            "type": "object",
            "properties": {
                "equipment_type":     {"type": "string", "description": "設備の種類"},
                "installation_place": {"type": "string", "description": "設置場所（建屋・階・部屋名）"},
                "operation_purpose":  {"type": "string", "description": "設備の用途・目的"},
                "scheduled_date":     {"type": "string", "description": "稼働開始予定日"},
                "chemicals":          {"type": "string", "description": "薬品・ガスの使用有無・種類"},
                "fire_exhaust":       {"type": "string", "description": "火気・排気の発生有無"},
                "wastewater":         {"type": "string", "description": "排水の発生有無"},
                "noise_vibration":    {"type": "string", "description": "騒音・振動の発生有無"},
                "radiation":          {"type": "string", "description": "放射線・X線装置への該当有無"},
                "construction":       {"type": "string", "description": "建屋改修・工事の有無"},
                "additional_info":    {"type": "string", "description": "上記10項目以外にユーザーが申告した追加情報。追加情報の質問をユーザーに送り、実際の回答を受け取ってからこのフィールドに記入すること。回答がない場合は「なし」と記入する。"},
            },
            "required": [
                "equipment_type", "installation_place", "operation_purpose", "scheduled_date",
                "chemicals", "fire_exhaust", "wastewater", "noise_vibration", "radiation", "construction",
                "additional_info",
            ],
        },
    },
}


# ─── ノード: ヒアリング ───────────────────────────────────────────
def hearing_node(state: AppState) -> dict:
    """GPT によるヒアリング。complete_hearing ツールが呼ばれたら次フェーズへ。"""
    llm = _llm()
    llm_with_tools = llm.bind_tools([COMPLETE_HEARING_TOOL])

    messages = [SystemMessage(HEARING_SYSTEM)] + list(state.get("messages", []))
    response = llm_with_tools.invoke(messages)

    updates: dict = {"messages": [response]}

    tool_calls = getattr(response, "tool_calls", None) or []
    for tc in tool_calls:
        if tc.get("name") == "complete_hearing":
            info = tc.get("args", {})
            tool_msg = ToolMessage(
                content="設備情報の収集が完了しました。分析を開始します。",
                tool_call_id=tc.get("id", ""),
            )
            updates.update({
                "messages": [response, tool_msg],
                "equipment_info": info,
                "hearing_complete": True,
                "phase": "analysis",
            })
            break

    return updates


def _route_hearing(state: AppState) -> Literal["analysis", "__end__"]:
    return "analysis" if state.get("hearing_complete") else END


# ─── ノード: 分析 ─────────────────────────────────────────────────
def analysis_node(state: AppState) -> dict:
    """設備情報を分析して論点整理。interrupt で方針レビューを要求。"""
    structured_llm = _llm().with_structured_output(AnalysisResult)
    info = state.get("equipment_info", {})
    info_str = json.dumps(info, ensure_ascii=False, indent=2)

    result: AnalysisResult = structured_llm.invoke([
        SystemMessage(ANALYSIS_SYSTEM),
        HumanMessage(f"以下の設備情報を分析してください:\n\n{info_str}"),
    ])

    # Human in the loop: 方針レビュー
    decision = interrupt({
        "phase":            "policy_review",
        "equipment_info":   info,
        "issues":           result.issues,
        "unknown_items":    result.unknown_items,
        "search_keywords":  result.search_keywords,
        "search_plan":      result.search_plan,
        "analysis_summary": result.analysis_summary,
    })

    # 再開後: 担当者の追記（「approved: <追記>」形式）を抽出して調査に反映する
    policy_note = ""
    if isinstance(decision, str):
        for prefix in ("approved:", "approved："):
            if decision.startswith(prefix):
                policy_note = decision.split(prefix, 1)[1].strip()
                break

    return {
        "messages":       [AIMessage("調査方針が承認されました。e-Gov API で法令を調査します...")],
        "issues":         result.issues,
        "unknown_items":  result.unknown_items,
        "search_keywords": result.search_keywords,
        "search_plan":    result.search_plan,
        "policy_note":    policy_note,
        "phase":          "searching",
    }


# ─── Web検索結果の処理ヘルパー ────────────────────────────────────
def _process_web_results(
    web_results: list[dict],
    query: str,
    seen_titles: set,
    results: list,
    label: str,
) -> tuple[str, int]:
    """Web検索結果を処理し、エラーと正常結果を判別してログエントリを返す。"""
    # エラー結果の検出
    errors = [r for r in web_results if r.get("source") == "error"]
    if errors:
        err_msg = errors[0].get("title", "不明なエラー")
        entry = f"⚠️ [{label}] 「{query}」→ エラー: {err_msg}"
        return entry, 0

    added = 0
    for r in web_results:
        key = r.get("title", "") + r.get("url", "")
        if key not in seen_titles:
            seen_titles.add(key)
            results.append(r)
            added += 1
    entry = f"🌐 [{label}] 「{query}」→ {added}件取得"
    return entry, added


# ─── ノード: 検索（Agentic Search） ──────────────────────────────
def search_node(state: AppState) -> dict:
    """LLMが検索結果を見ながら次のクエリを自律的に判断するAgenticループ。"""
    search_llm = _llm().with_structured_output(SearchAction)

    equipment_info   = state.get("equipment_info", {})
    issues           = state.get("issues", [])
    initial_keywords = state.get("search_keywords", [])
    policy_note      = state.get("policy_note", "")

    # 再調査ループ時は前回の検索結果を引き継ぎ、新規分を上乗せする
    prev_results: list[dict] = state.get("search_results", []) or []
    results: list[dict] = list(prev_results)
    seen_titles: set[str] = set()
    for _r in prev_results:
        _t = _r.get("title", "")
        if _t:
            seen_titles.add(_t)
            seen_titles.add(_t + _r.get("url", ""))
    search_log: list[str] = []
    progress_messages: list = []
    failed_queries: set[str] = set()  # 0件だったクエリを記録

    # 進捗のライブ送信（stream_mode="custom"）。invoke 実行時は no-op。
    _writer = None
    if get_stream_writer is not None:
        try:
            _writer = get_stream_writer()
        except Exception:
            _writer = None

    def emit(entry: str) -> None:
        if _writer is not None:
            try:
                _writer({"progress": entry})
            except Exception:
                pass

    emit("🔎 法令調査を開始します（e-Gov 法令API ＋ Web検索）...")

    # シード: 最初の3キーワードで初期データを収集してからLLMに判断させる
    emit("📚 e-Gov 法令API でキーワード検索中...")
    for kw in initial_keywords[:3]:
        laws = search_laws_by_keyword(kw, max_results=4)
        added = 0
        for law in laws:
            title = law.get("title", "")
            if title and title not in seen_titles:
                seen_titles.add(title)
                results.append(law)
                added += 1
        entry = f"🔍 [e-Gov] 「{kw}」→ {added}件取得"
        search_log.append(entry)
        progress_messages.append(AIMessage(content=entry, name="search_progress"))
        emit(entry)

    # 必須Web検索①: 横浜市・神奈川県の条例・届出施設
    emit("🌐 横浜市・神奈川県の条例・届出施設をWeb検索中...")
    equipment_type_str = equipment_info.get("equipment_type", "設備")
    for local_query in [
        f"横浜市 {equipment_type_str} 届出施設 手続き 規制",
        f"横浜市 {equipment_type_str} 届出 条例 規制",
        f"神奈川県 {equipment_type_str} 届出 条例 規制",
    ]:
        web_results = search_web(local_query)
        entry, added = _process_web_results(web_results, local_query, seen_titles, results, "条例Web")
        search_log.append(entry)
        progress_messages.append(AIMessage(content=entry, name="search_progress"))
        emit(entry)

    # 必須Web検索②: 省庁ガイドライン・FAQ
    emit("🌐 省庁ガイドライン・FAQ をWeb検索中...")
    guideline_query = f"{equipment_type_str} 設置 届出 省庁 ガイドライン FAQ"
    web_results = search_web(guideline_query)
    entry, added = _process_web_results(web_results, guideline_query, seen_titles, results, "ガイドラインWeb")
    search_log.append(entry)
    progress_messages.append(AIMessage(content=entry, name="search_progress"))
    emit(entry)

    # Agenticループ
    for iteration in range(MAX_SEARCH_ITERATIONS):
        # 進捗表示：何回目の判断か
        progress_msg = f"🔄 AIが追加調査の要否を判断中（{iteration + 1}回目／最大{MAX_SEARCH_ITERATIONS}回）"
        progress_messages.append(AIMessage(content=progress_msg, name="search_progress"))
        emit(progress_msg)

        issues_str   = "\n".join(f"- {i}" for i in issues)
        keywords_str = "\n".join(f"- {k}" for k in initial_keywords)
        history_str  = "\n".join(search_log) if search_log else "（なし）"
        results_str  = "\n".join(
            f"- {r.get('title','?')} ({r.get('source','?')})"
            + (f"\n    概要: {r.get('snippet','')[:150]}" if r.get('snippet') else "")
            for r in results[:25]
        ) if results else "（なし）"

        note_str = (
            f"\n\n## 🔔 担当者からの追記指示（最優先で考慮し、関連する検索を追加すること）\n{policy_note}"
            if policy_note else ""
        )

        failed_str = "\n".join(f"- {q}" for q in failed_queries) if failed_queries else "なし"
        context = (
            f"## 設備情報\n{json.dumps(equipment_info, ensure_ascii=False, indent=2)}\n\n"
            f"## 調査が必要な項目\n{issues_str}\n\n"
            f"## 推奨検索キーワード（参考）\n{keywords_str}\n\n"
            f"## 検索履歴（{iteration + 1}回目判断）\n{history_str}\n\n"
            f"## ⚠️ 0件だったクエリ（再検索禁止）\n{failed_str}\n\n"
            f"## 取得済み法令・情報（{len(results)}件）\n{results_str}"
            f"{note_str}"
        )

        action: SearchAction = search_llm.invoke([
            SystemMessage(SEARCH_AGENT_SYSTEM),
            HumanMessage(context),
        ])

        if action.done or not action.query:
            done_entry = f"✅ 調査完了（{iteration + 1}回の判断で収集十分と判断しました）"
            progress_messages.append(AIMessage(content=done_entry, name="search_progress"))
            emit(done_entry)
            break

        # 0件クエリの再検索をスキップ
        if action.query in failed_queries:
            skip_entry = f"⏭️ スキップ（0件クエリ再試行）:「{action.query}」"
            search_log.append(skip_entry)
            progress_messages.append(AIMessage(content=skip_entry, name="search_progress"))
            emit(skip_entry)
        elif action.search_type == "egov":
            emit(f"📚 e-Gov 法令API で追加検索中:「{action.query}」")
            laws = search_laws_by_keyword(action.query, max_results=5)
            added = 0
            for law in laws:
                title = law.get("title", "")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    results.append(law)
                    added += 1
            if added == 0:
                failed_queries.add(action.query)
            entry = f"🔍 [e-Gov API] 「{action.query}」→ {added}件新規取得"
            search_log.append(entry)
            progress_messages.append(AIMessage(content=entry, name="search_progress"))
            emit(entry)
        else:
            emit(f"🌐 Web検索中:「{action.query}」")
            web_results = search_web(action.query)
            entry, added = _process_web_results(web_results, action.query, seen_titles, results, "AIWeb検索")
            entry = entry.replace("取得", "新規取得")
            if added == 0:
                failed_queries.add(action.query)
            search_log.append(entry)
            progress_messages.append(AIMessage(content=entry, name="search_progress"))
            emit(entry)

    egov_count = len([r for r in results if "e-Gov" in r.get("source", "")])
    web_count  = len([r for r in results if "e-Gov" not in r.get("source", "")])
    summary_msg = (
        f"調査完了（{len(search_log)}回検索）。"
        f"e-Gov {egov_count}件、Web {web_count}件、計{len(results)}件を収集しました。"
    )
    progress_messages.append(AIMessage(content=summary_msg))
    emit(summary_msg)
    emit("🧩 調査結果を統合し、法令別の対応事項を整理中...")

    return {
        "search_results": results[:30],
        "phase":          "synthesizing",
        "messages":       progress_messages,
    }


# ─── ノード: 結果統合 ─────────────────────────────────────────────
def synthesis_node(state: AppState) -> dict:
    """検索結果を統合してアクションアイテムを生成。interrupt で結果レビューを要求。"""
    structured_llm = _llm(max_tokens=16000).with_structured_output(SynthesisResult)

    equipment_info = state.get("equipment_info", {})
    search_results = state.get("search_results", [])
    issues         = state.get("issues", [])
    policy_note    = state.get("policy_note", "")

    context = (
        f"【設備情報】\n{json.dumps(equipment_info, ensure_ascii=False, indent=2)}\n\n"
        f"【確認が必要な論点】\n{json.dumps(issues, ensure_ascii=False)}\n\n"
        + (f"【担当者からの追記指示（最優先で反映）】\n{policy_note}\n\n" if policy_note else "")
        + f"【e-Gov・Web 調査結果（抜粋）】\n"
        + "\n".join(
            f"- {r.get('title','?')} ({r.get('source','?')})"
            + (f"\n  概要: {r.get('snippet','')[:300]}" if r.get('snippet') else "")
            for r in search_results[:18]
        )
    )

    result: SynthesisResult = structured_llm.invoke([
        SystemMessage(SYNTHESIS_SYSTEM),
        HumanMessage(context),
    ])

    law_items = [item.model_dump() for item in result.law_items]

    # e-Gov の law_id / law_revision_id を法令名で逆引きして付与
    search_results = state.get("search_results", [])
    title_to_ids: dict[str, dict] = {}
    for r in search_results:
        t = r.get("title", "")
        if t and r.get("source", "").startswith("e-Gov"):
            title_to_ids[t] = {
                "law_id":          r.get("law_id", ""),
                "law_revision_id": r.get("law_revision_id", ""),
            }
    for law in law_items:
        law_name = law.get("law_name", "")
        for title, ids in title_to_ids.items():
            if law_name in title or title in law_name:
                law["law_id"]          = ids["law_id"]
                law["law_revision_id"] = ids["law_revision_id"]
                break
        else:
            law.setdefault("law_id", "")
            law.setdefault("law_revision_id", "")

    # law_id が未解決の法令だけ e-Gov API で補完検索
    for law in law_items:
        if not law.get("law_id"):
            law_name = law.get("law_name", "")
            if law_name:
                hits = search_laws_by_keyword(law_name, max_results=1)
                if hits:
                    law["law_id"] = hits[0].get("law_id", "")

    # relevant_articles が空なら deliveries の law_article から補完
    for law in law_items:
        if not law.get("relevant_articles"):
            arts = [
                d.get("law_article", "")
                for d in law.get("deliveries", [])
                if d.get("law_article", "")
            ]
            if arts:
                law["relevant_articles"] = list(dict.fromkeys(arts))

    # Human in the loop: 結果レビュー
    decision = interrupt({
        "phase":      "results_review",
        "law_items":  law_items,
        "summary":    result.summary,
        "risk_count": result.risk_count.model_dump(),
    })

    # 再開後: 「足りない・再調査して」依頼なら検索フェーズに戻す
    if _is_reinvestigate(decision):
        extra = _extract_after(decision, _REINVEST_PREFIXES)
        return {
            "messages":   [AIMessage(f"担当者から追加調査の依頼を受けました：{extra}\n再調査を実施します...")],
            "law_items":  law_items,
            "policy_note": _merge_note(state.get("policy_note", ""), extra),
            "issues":     list(state.get("issues", []) or []) + [f"【追加調査依頼】{extra}"],
            "phase":      "searching",
        }

    # 通常: レポートへ。case_id をここで確定し、以降の再実行でも不変にする
    case_id = state.get("case_id") or (
        f"EQ-{datetime.datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
    )
    return {
        "messages":  [AIMessage("レビュー完了。レポートを生成します...")],
        "law_items": law_items,
        "case_id":   case_id,
        "phase":     "reporting",
    }


# ─── 文面修正（LLM）ヘルパー ──────────────────────────────────────
def _refine_law_items(law_items: list, equipment_info: dict, instruction: str) -> list:
    """担当者の文面修正依頼に沿って law_items のテキストを LLM で調整する。
    法令名・条番号・届出先などの事実情報は依頼が無い限り変えない。
    失敗時は元の law_items を返す。"""
    if not law_items:
        return law_items
    refine_llm = _llm(max_tokens=16000).with_structured_output(RefineResult)
    ctx = (
        f"【設備情報】\n{json.dumps(equipment_info, ensure_ascii=False, indent=2)}\n\n"
        f"【現在の法令別対応事項(JSON)】\n{json.dumps(law_items, ensure_ascii=False, indent=2)}\n\n"
        f"【担当者からの文面修正依頼】\n{instruction}\n\n"
        "上記の依頼に沿って law_items の文面（applicability / item / deadline / responsible など）"
        "を読みやすく修正してください。法令名・条番号(relevant_articles, law_article)・"
        "届出先(authority)・priority などの事実情報は、依頼で明示されない限り変更しないこと。"
    )
    try:
        res: RefineResult = refine_llm.invoke([SystemMessage(SYNTHESIS_SYSTEM), HumanMessage(ctx)])
        refined = [it.model_dump() for it in res.law_items]
    except Exception:
        return law_items

    # LawItem スキーマに無い law_id / law_revision_id を法令名で再付与（条文リンク維持）
    orig_by_name = {l.get("law_name", ""): l for l in law_items}
    for it in refined:
        o = orig_by_name.get(it.get("law_name", ""))
        if o:
            it["law_id"]          = o.get("law_id", "")
            it["law_revision_id"] = o.get("law_revision_id", "")
    return refined or law_items


# ─── ノード: レポート生成 ──────────────────────────────────────────
def report_node(state: AppState) -> dict:
    """HTML レポートを生成。interrupt でレポートレビューを要求。
    再開時の依頼に応じて 再調査(search へ) / 文面修正(report 再実行) / 承認(complete) に分岐する。"""
    case_id = state.get("case_id") or (
        f"EQ-{datetime.datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
    )
    report_html = generate_html_report(
        equipment_info=state.get("equipment_info", {}),
        law_items=state.get("law_items", []),
        unknown_items=state.get("unknown_items", []),
        search_results=state.get("search_results", []),
        case_id=case_id,
    )

    # Human in the loop: レポートレビュー
    decision = interrupt({
        "phase":       "report_review",
        "report_html": report_html,
        "case_id":     case_id,
    })

    # ① 「足りない・再調査して」→ 検索フェーズに戻す
    if _is_reinvestigate(decision):
        extra = _extract_after(decision, _REINVEST_PREFIXES)
        return {
            "messages":    [AIMessage(f"レポート確認後、追加調査の依頼を受けました：{extra}\n再調査を実施します...")],
            "policy_note": _merge_note(state.get("policy_note", ""), extra),
            "issues":      list(state.get("issues", []) or []) + [f"【追加調査依頼】{extra}"],
            "case_id":     case_id,
            "phase":       "searching",
        }

    # ② 文面修正（LLM）→ law_items を更新し report を再実行して再確認させる
    refine = _extract_after(decision, _REFINE_PREFIXES)
    if refine:
        new_items = _refine_law_items(
            state.get("law_items", []), state.get("equipment_info", {}), refine
        )
        return {
            "messages":  [AIMessage(f"レポート文面の修正依頼を受けました：{refine}\n修正して再生成します...")],
            "law_items": new_items,
            "case_id":   case_id,
            "phase":     "reporting",
        }

    # ③ 承認 → 完了
    return {
        "messages": [AIMessage(
            "✅ レポートが承認されました。\n\n"
            "設備導入時の法令・手続き確認が完了しました。\n"
            "レポートをダウンロードして、担当部署に共有してください。"
        )],
        "report_html": report_html,
        "case_id":     case_id,
        "phase":       "complete",
    }


# ─── グラフ構築 ───────────────────────────────────────────────────
def _route_start(state: AppState) -> str:
    return {
        "hearing":      "hearing",
        "analysis":     "analysis",
        "searching":    "search",
        "synthesizing": "synthesis",
        "reporting":    "report",
    }.get(state.get("phase", "hearing"), "hearing")


def _route_after_synthesis(state: AppState) -> str:
    """結果確認の後段ルーティング。再調査依頼なら search、通常は report。"""
    return "search" if state.get("phase") == "searching" else "report"


def _route_after_report(state: AppState):
    """レポート確認の後段ルーティング。"""
    phase = state.get("phase", "complete")
    if phase == "searching":
        return "search"
    if phase == "reporting":
        return "report"
    return END


def build_workflow():
    builder = StateGraph(AppState)

    builder.add_node("hearing",   hearing_node)
    builder.add_node("analysis",  analysis_node)
    builder.add_node("search",    search_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("report",    report_node)

    builder.add_conditional_edges(START, _route_start, {
        "hearing":   "hearing",
        "analysis":  "analysis",
        "search":    "search",
        "synthesis": "synthesis",
        "report":    "report",
    })
    builder.add_conditional_edges("hearing", _route_hearing, {
        "analysis": "analysis",
        END: END,
    })
    builder.add_edge("analysis",  "search")
    builder.add_edge("search",    "synthesis")
    # 結果確認で「再調査」依頼 → search に戻す。それ以外は report へ
    builder.add_conditional_edges("synthesis", _route_after_synthesis, {
        "search": "search",
        "report": "report",
    })
    # レポート確認で 再調査 → search / 文面修正 → report再実行 / 承認 → END
    builder.add_conditional_edges("report", _route_after_report, {
        "search": "search",
        "report": "report",
        END:      END,
    })

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# シングルトン
workflow = build_workflow()


# ─── ユーティリティ ───────────────────────────────────────────────
def get_interrupt_data(thread_id: str) -> dict | None:
    """現在の interrupt データを取得する。なければ None。"""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = workflow.get_state(config)
        if not state.next:
            return None
        for task in state.tasks:
            interrupts = getattr(task, "interrupts", [])
            if interrupts:
                return interrupts[0].value
    except Exception:
        pass
    return None


def get_all_messages(thread_id: str) -> list:
    """チェックポイントから全メッセージを取得する。"""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = workflow.get_state(config)
        return list(state.values.get("messages", []))
    except Exception:
        return []


def get_state_value(thread_id: str, key: str):
    """チェックポイントから特定のキーの値を取得する。"""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = workflow.get_state(config)
        return state.values.get(key)
    except Exception:
        return None
