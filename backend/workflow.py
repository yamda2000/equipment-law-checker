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


MAX_SEARCH_ITERATIONS = 10


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
    interrupt({
        "phase":            "policy_review",
        "equipment_info":   info,
        "issues":           result.issues,
        "unknown_items":    result.unknown_items,
        "search_keywords":  result.search_keywords,
        "search_plan":      result.search_plan,
        "analysis_summary": result.analysis_summary,
    })

    # 再開後
    return {
        "messages":       [AIMessage("調査方針が承認されました。e-Gov API で法令を調査します...")],
        "issues":         result.issues,
        "unknown_items":  result.unknown_items,
        "search_keywords": result.search_keywords,
        "search_plan":    result.search_plan,
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

    results: list[dict] = []
    seen_titles: set[str] = set()
    search_log: list[str] = []
    progress_messages: list = []

    # シード: 最初の3キーワードで初期データを収集してからLLMに判断させる
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

    # 必須Web検索①: 横浜市・神奈川県の条例
    equipment_type_str = equipment_info.get("equipment_type", "設備")
    for local_query in [
        f"横浜市 {equipment_type_str} 届出 条例 規制",
        f"神奈川県 {equipment_type_str} 届出 条例 規制",
    ]:
        web_results = search_web(local_query)
        entry, added = _process_web_results(web_results, local_query, seen_titles, results, "条例Web")
        search_log.append(entry)
        progress_messages.append(AIMessage(content=entry, name="search_progress"))

    # 必須Web検索②: 省庁ガイドライン・FAQ
    guideline_query = f"{equipment_type_str} 設置 届出 省庁 ガイドライン FAQ"
    web_results = search_web(guideline_query)
    entry, added = _process_web_results(web_results, guideline_query, seen_titles, results, "ガイドラインWeb")
    search_log.append(entry)
    progress_messages.append(AIMessage(content=entry, name="search_progress"))

    # Agenticループ
    for iteration in range(MAX_SEARCH_ITERATIONS):
        issues_str   = "\n".join(f"- {i}" for i in issues)
        keywords_str = "\n".join(f"- {k}" for k in initial_keywords)
        history_str  = "\n".join(search_log) if search_log else "（なし）"
        results_str  = "\n".join(
            f"- {r.get('title','?')} ({r.get('source','?')})"
            for r in results[:25]
        ) if results else "（なし）"

        context = (
            f"## 設備情報\n{json.dumps(equipment_info, ensure_ascii=False, indent=2)}\n\n"
            f"## 調査が必要な項目\n{issues_str}\n\n"
            f"## 推奨検索キーワード（参考）\n{keywords_str}\n\n"
            f"## 検索履歴（{iteration + 1}回目判断）\n{history_str}\n\n"
            f"## 取得済み法令・情報（{len(results)}件）\n{results_str}"
        )

        action: SearchAction = search_llm.invoke([
            SystemMessage(SEARCH_AGENT_SYSTEM),
            HumanMessage(context),
        ])

        if action.done or not action.query:
            break

        if action.search_type == "egov":
            laws = search_laws_by_keyword(action.query, max_results=5)
            added = 0
            for law in laws:
                title = law.get("title", "")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    results.append(law)
                    added += 1
            entry = f"🔍 [e-Gov] 「{action.query}」→ {added}件新規取得"
            search_log.append(entry)
            progress_messages.append(AIMessage(content=entry, name="search_progress"))
        else:
            web_results = search_web(action.query)
            entry, added = _process_web_results(web_results, action.query, seen_titles, results, "Web")
            entry = entry.replace("取得", "新規取得")
            search_log.append(entry)
            progress_messages.append(AIMessage(content=entry, name="search_progress"))

    egov_count = len([r for r in results if "e-Gov" in r.get("source", "")])
    web_count  = len([r for r in results if "e-Gov" not in r.get("source", "")])
    summary_msg = (
        f"調査完了（{len(search_log)}回検索）。"
        f"e-Gov {egov_count}件、Web {web_count}件、計{len(results)}件を収集しました。"
    )
    progress_messages.append(AIMessage(content=summary_msg))

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

    context = (
        f"【設備情報】\n{json.dumps(equipment_info, ensure_ascii=False, indent=2)}\n\n"
        f"【確認が必要な論点】\n{json.dumps(issues, ensure_ascii=False)}\n\n"
        f"【e-Gov・Web 調査結果（抜粋）】\n"
        + "\n".join(
            f"- {r.get('title','?')} ({r.get('source','?')})"
            for r in search_results[:15]
        )
    )

    result: SynthesisResult = structured_llm.invoke([
        SystemMessage(SYNTHESIS_SYSTEM),
        HumanMessage(context),
    ])

    law_items = [item.model_dump() for item in result.law_items]

    # e-Gov の law_id を法令名で逆引きして付与
    search_results = state.get("search_results", [])
    title_to_id = {
        r["title"]: r["law_id"]
        for r in search_results
        if r.get("law_id") and r.get("title") and r.get("source", "").startswith("e-Gov")
    }
    for law in law_items:
        law_name = law.get("law_name", "")
        for title, lid in title_to_id.items():
            if law_name in title or title in law_name:
                law["law_id"] = lid
                break
        else:
            law.setdefault("law_id", "")

    # Human in the loop: 結果レビュー
    interrupt({
        "phase":      "results_review",
        "law_items":  law_items,
        "summary":    result.summary,
        "risk_count": result.risk_count.model_dump(),
    })

    return {
        "messages":  [AIMessage("レビュー完了。レポートを生成します...")],
        "law_items": law_items,
        "phase":     "reporting",
    }


# ─── ノード: レポート生成 ──────────────────────────────────────────
def report_node(state: AppState) -> dict:
    """HTML レポートを生成。interrupt でレポートレビューを要求。"""
    case_id = f"EQ-{datetime.datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
    report_html = generate_html_report(
        equipment_info=state.get("equipment_info", {}),
        law_items=state.get("law_items", []),
        unknown_items=state.get("unknown_items", []),
        search_results=state.get("search_results", []),
        case_id=case_id,
    )

    # Human in the loop: レポートレビュー
    interrupt({
        "phase":       "report_review",
        "report_html": report_html,
        "case_id":     case_id,
    })

    return {
        "messages": [AIMessage(
            "✅ レポートが承認されました。\n\n"
            "設備導入時の法令・手続き確認が完了しました。\n"
            "レポートをダウンロードして、担当部署に共有してください。"
        )],
        "report_html": report_html,
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
    builder.add_edge("synthesis", "report")
    builder.add_edge("report",    END)

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
