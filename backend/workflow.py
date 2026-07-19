"""
LangGraph ワークフロー
フェーズ: hearing → analysis → search → synthesis → report → complete
Human in the loop: analysis / synthesis / report の3箇所で interrupt
LLM: PoC=OpenAI API / 本番=Azure OpenAI (LLM_MODE 環境変数で切り替え)
"""

import os
import re
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
from backend.prompts import (
    HEARING_SYSTEM, ANALYSIS_SYSTEM, SYNTHESIS_SYSTEM, SEARCH_AGENT_SYSTEM,
    ARTICLE_SELECTION_SYSTEM, PRE_SYNTHESIS_ARTICLE_SYSTEM, COVERAGE_CHECK_SYSTEM,
    SUPPLEMENTAL_LAW_SYSTEM,
)
from backend.tools.egov import (
    search_laws_by_keyword, fetch_article_list, fetch_article_text,
    normalize_article_ref, get_suggested_keywords, resolve_law_alias,
)
from backend.tools.web_search import search_web
from backend.tools.internal_docs import search_internal_docs, internal_docs_available
from backend.rag_agent import agentic_internal_search
from backend.case_memory import save_case, find_similar_cases, format_cases_for_prompt
from backend.fields import FIELD_JA as HEARING_FIELD_JA
from backend.report_gen import generate_html_report


# ─── LLM ファクトリ ───────────────────────────────────────────────
def _llm(temperature: float = 1.0, max_tokens: int | None = None):
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


MAX_SEARCH_ITERATIONS = 30    # Agenticループの最大検索回数
MAX_SEARCH_RESULTS    = 120   # 収集結果の保持・統合入力の上限（漏れ防止のため広めに取る）


class SearchAction(BaseModel):
    done: bool = Field(description="True=収集完了, False=追加検索が必要")
    search_type: Literal["egov", "web", "internal"] = Field(
        default="egov",
        description="egov: e-Gov法令API / web: Web検索 / internal: 社内文書検索（登録済みの場合のみ）",
    )
    query: str = Field(default="", description="次の検索クエリ（done=Falseの場合は必須）")
    reason: str = Field(description="判断理由")


class UncoveredIssue(BaseModel):
    issue: str = Field(description="対応する法令・情報がまだ見つかっていない論点（入力の論点リストの表記のまま）")
    suggested_query: str = Field(default="", description="この論点を埋めるための追加検索クエリ（egovの場合は法令名のみ）")
    search_type: Literal["egov", "web", "internal"] = Field(
        default="egov",
        description="egov: e-Gov法令API / web: Web検索 / internal: 社内文書検索（登録済みの場合のみ）",
    )


class CoveredIssue(BaseModel):
    issue: str = Field(description="カバー済みの論点（入力の論点リストの表記のまま）")
    covered_by: list[str] = Field(
        default_factory=list,
        description="この論点をカバーする収集済み法令・情報のタイトル（一覧の表記のまま・最大3件）",
    )


class CoverageCheck(BaseModel):
    uncovered: list[UncoveredIssue] = Field(
        default_factory=list,
        description="対応情報が見つかっていない論点リスト。全論点カバー済みなら空リスト",
    )
    covered: list[CoveredIssue] = Field(
        default_factory=list,
        description="カバー済みの論点と、その根拠となった収集済み法令・情報のタイトル",
    )


class DeliveryItem(BaseModel):
    item: str = Field(description="届出・申請の名称")
    authority: str = Field(description="届出先・申請先機関")
    authority_basis: str = Field(
        default="",
        description=(
            "届出先の根拠（1〜2行）。法令上の届出先規定（例：騒音規制法第6条「市町村長に届け出」）と、"
            "実際の窓口の出典（例：横浜市公式サイトの案内）を書く。"
            "条文抜粋・Web検索結果に根拠がなく推定の場合は「一般的な所管に基づく推定（要確認）」と明記する"
        ),
    )
    authority_source_id: str = Field(
        default="",
        description=(
            "届出先の根拠に使った調査結果のID（調査結果一覧の [R◯] の◯部分を含む形式。例：\"R12\"）。"
            "Webページ等を出典にした場合のみ記入し、条文のみが根拠・推定の場合は空文字"
        ),
    )
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


class ExcludedLaw(BaseModel):
    law_name: str = Field(description="確認したうえで非該当と判断した法令名・領域")
    reason: str = Field(description="非該当と判断した理由（設備情報のどの回答に基づくか明示）")
    basis: Literal["confirmed", "insufficient_info"] = Field(
        default="confirmed",
        description=(
            "confirmed: 設備情報の回答に基づき非該当と確認済み / "
            "insufficient_info: 判断材料が不足しており非該当と断定できない（要確認扱いになる）"
        ),
    )


class AdditionalCheckLaw(BaseModel):
    law_name: str = Field(description="調査結果一覧に根拠情報はないが、一般知識から適用の可能性が否定できない法令名（正式名称）")
    reason: str = Field(description="設備情報のどの点から適用の可能性があると考えるか（1〜2文）")


class SynthesisResult(BaseModel):
    law_items: list[LawItem] = Field(description="法令別の対応事項リスト")
    summary: str = Field(description="調査結果の総括（3〜5行）")
    risk_count: RiskCount = Field(description="優先度別の法令件数")
    excluded_laws: list[ExcludedLaw] = Field(
        default_factory=list,
        description="主要法令領域のうち、確認したが非該当と判断した法令と理由のリスト",
    )
    additional_check_laws: list[AdditionalCheckLaw] = Field(
        default_factory=list,
        description="調査結果に根拠はないが適用の可能性が否定できない追加調査候補（0〜5件）",
    )


class SupplementalJudgment(BaseModel):
    law_name: str = Field(description="判定対象の法令名（候補と同じ名称）")
    verdict: Literal["check", "excluded"] = Field(
        description="check: 適用の可能性が否定できない（要確認） / excluded: 設備情報から明確に非該当"
    )
    reason: str = Field(description="判定理由（設備情報のどの回答に基づくか）")


class SupplementalResult(BaseModel):
    judgments: list[SupplementalJudgment] = Field(
        default_factory=list,
        description="候補法令全件分の判定リスト",
    )


class RefineResult(BaseModel):
    law_items: list[LawItem] = Field(description="文面修正後の法令別対応事項リスト")


class ArticleSelection(BaseModel):
    articles: list[str] = Field(
        default_factory=list,
        description="選択した条番号リスト（提示された条文一覧に実在するもののみ・最大5件）",
    )


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


def _split_attachments(extra: str) -> tuple:
    """依頼文から添付資料の抜粋部分（UI側が「【添付資料「…」の内容抜粋】」形式で
    連結する）を切り離す。抜粋は数千字になり得るため、チャット表示や論点リストには
    依頼文だけを使い、全文は policy_note（調査プロンプト）にのみ渡す。
    戻り値: (依頼文のみ, 添付有無)"""
    i = extra.find("【添付資料「")
    if i == -1:
        return extra.strip(), False
    return extra[:i].strip(), True


# ─── ヒアリングツール定義（OpenAI/Azure 形式）────────────────────
COMPLETE_HEARING_TOOL = {
    "type": "function",
    "function": {
        "name": "complete_hearing",
        "description": (
            "設備情報の収集が完了した場合に呼び出す。"
            "全項目が「資料からの確定情報」または「ユーザーへの質問で得た回答（不明・未定を含む）」で埋まったときのみ使用する。"
            "「（未記入）」のまま質問していない項目が1つでも残っている間は絶対に呼び出してはいけない。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "equipment_type":     {"type": "string", "description": "設備の種類"},
                "installation_place": {"type": "string", "description": "設置場所（建屋・階・部屋名）"},
                "operation_purpose":  {"type": "string", "description": "設備の用途・目的"},
                "scheduled_date":     {"type": "string", "description": "稼働開始予定日"},
                "chemicals":          {"type": "string", "description": "薬品・溶剤・ガス・燃料（冷媒・圧縮ガス含む）の使用有無・種類・使用量・貯蔵量"},
                "fire_exhaust":       {"type": "string", "description": "火気・熱源・排気・粉じんの発生有無"},
                "wastewater":         {"type": "string", "description": "排水・廃液・廃棄物の発生有無"},
                "noise_vibration":    {"type": "string", "description": "騒音・振動の発生有無"},
                "radiation":          {"type": "string", "description": "放射線・X線装置への該当有無"},
                "construction":       {"type": "string", "description": "建屋改修・電気工事（受電容量増加・自家発電機・蓄電池含む）・配管工事の有無"},
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

# 完了時に ToolMessage へ入れる文言。app.py 側がこのマーカーで
# 「ヒアリング完了」表示を判定するため、変更時は app.py と揃えること
HEARING_COMPLETE_MARKER = "設備情報の収集が完了しました。分析を開始します。"


def _unanswered_fields(info: dict) -> list[str]:
    """complete_hearing の引数のうち、質問せずに放置された項目を返す。
    空文字・「（未記入）」はユーザーの回答ではない（「不明」「未定」は回答として有効）。"""
    return [
        k for k in HEARING_FIELD_JA
        if not str(info.get(k, "") or "").strip()
        or "未記入" in str(info.get(k, ""))
    ]


def hearing_node(state: AppState) -> dict:
    """GPT によるヒアリング。complete_hearing ツールが呼ばれたら次フェーズへ。

    ただし「（未記入）」のまま質問していない項目が残る complete_hearing は却下し、
    質問を続けさせる（資料アップロード時、LLMが（未記入）を回答済みと誤解して
    1問だけで完了してしまう事象への決定的ガード）。"""
    llm = _llm()
    llm_with_tools = llm.bind_tools([COMPLETE_HEARING_TOOL])

    base_messages = [SystemMessage(HEARING_SYSTEM)] + list(state.get("messages", []))
    new_messages: list = []

    max_attempts = 3
    for attempt in range(max_attempts):
        response = llm_with_tools.invoke(base_messages + new_messages)
        new_messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        tc = next((t for t in tool_calls if t.get("name") == "complete_hearing"), None)
        if tc is None:
            # 通常の質問メッセージ。ユーザーの回答を待つ
            return {"messages": new_messages}

        info = tc.get("args", {})
        unanswered = _unanswered_fields(info)
        if unanswered and attempt < max_attempts - 1:
            # 未質問の項目が残っている → 完了を却下し、質問を続けるよう差し戻す
            ja_list = "、".join(HEARING_FIELD_JA[k] for k in unanswered)
            new_messages.append(ToolMessage(
                content=(
                    f"完了できません。次の項目がまだ「（未記入）」のままです：{ja_list}\n"
                    "「（未記入）」はユーザーの回答ではありません。complete_hearing を呼ばず、"
                    "上記の項目を1項目ずつユーザーに質問してください。"
                    "ユーザーから「不明」「未定」等の回答を得た場合のみ、その回答を記録できます。"
                ),
                tool_call_id=tc.get("id", ""),
            ))
            continue

        # 完了を受理（却下上限に達した場合は不完全でも先に進め、デッドロックを防ぐ）
        # 未記入のまま残った項目は「不明（未確認）」に正規化し、
        # 統合フェーズの「情報不足→要確認」ガードに乗せる
        for k in unanswered:
            info[k] = "不明（未確認）"
        new_messages.append(ToolMessage(
            content=HEARING_COMPLETE_MARKER,
            tool_call_id=tc.get("id", ""),
        ))
        return {
            "messages":         new_messages,
            "equipment_info":   info,
            "hearing_complete": True,
            "phase":            "analysis",
        }

    return {"messages": new_messages}


def _route_hearing(state: AppState) -> Literal["analysis", "__end__"]:
    return "analysis" if state.get("hearing_complete") else END


# ─── ノード: 分析 ─────────────────────────────────────────────────
def analysis_node(state: AppState) -> dict:
    """設備情報を分析して論点整理し、結果を state に保存する。
    方針レビュー（interrupt）は後続の policy_review_node で行う。
    interrupt を同一ノード内に置くと resume 時にノード全体（分析LLM）が
    再実行され、承認した方針と検索に使う方針がズレる・コストが倍になるため分離。"""
    structured_llm = _llm().with_structured_output(AnalysisResult)
    info = state.get("equipment_info", {})
    info_str = json.dumps(info, ensure_ascii=False, indent=2)

    # ケースメモリ（CBR）: 類似の承認済み過去案件を想起して分析の参考にする
    cases_str = ""
    try:
        similar = find_similar_cases(info, k=2, exclude_case_id=state.get("case_id", "") or "")
        cases_str = format_cases_for_prompt(similar)
    except Exception:
        cases_str = ""

    human = f"以下の設備情報を分析してください:\n\n{info_str}"
    if cases_str:
        human += (
            "\n\n## 🧠 類似の過去案件（担当者承認済みの実績・参考情報）\n"
            + cases_str
        )

    result: AnalysisResult = structured_llm.invoke([
        SystemMessage(ANALYSIS_SYSTEM),
        HumanMessage(human),
    ])

    updates: dict = {
        "issues":           result.issues,
        "unknown_items":    result.unknown_items,
        "search_keywords":  result.search_keywords,
        "search_plan":      result.search_plan,
        "analysis_summary": result.analysis_summary,
        "phase":            "policy_review",
    }
    if cases_str:
        n = cases_str.count("### 類似案件")
        updates["messages"] = [AIMessage(
            f"📚 過去の調査事例から類似案件 {n}件を参照して分析しました。"
        )]
    return updates


# ─── ノード: 方針レビュー（Human in the loop） ────────────────────
def policy_review_node(state: AppState) -> dict:
    """調査方針のレビュー。ノード先頭で interrupt するため、再開時に
    再実行されるのはこのノードだけで、state の分析結果は変わらない。"""
    decision = interrupt({
        "phase":            "policy_review",
        "equipment_info":   state.get("equipment_info", {}),
        "issues":           state.get("issues", []),
        "unknown_items":    state.get("unknown_items", []),
        "search_keywords":  state.get("search_keywords", []),
        "search_plan":      state.get("search_plan", ""),
        "analysis_summary": state.get("analysis_summary", ""),
    })

    # 担当者の追記（「approved: <追記>」形式）を抽出して調査に反映する
    policy_note = ""
    if isinstance(decision, str):
        for prefix in ("approved:", "approved："):
            if decision.startswith(prefix):
                policy_note = decision.split(prefix, 1)[1].strip()
                break

    return {
        "messages":    [AIMessage("調査方針が承認されました。e-Gov API で法令を調査します。")],
        "policy_note": policy_note,
        "phase":       "searching",
    }


# ─── 進捗ライブ送信ヘルパー ───────────────────────────────────────
def _make_emit():
    """stream_mode="custom" への進捗送信関数を返す。invoke 実行時は no-op。"""
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

    return emit


# ─── Web検索結果の処理ヘルパー ────────────────────────────────────
def _process_web_results(
    web_results: list[dict],
    query: str,
    seen_titles: set,
    results: list,
    label: str,
    icon: str = "🌐",
    suggestions_out: list | None = None,
) -> tuple[str, int]:
    """Web・社内文書の検索結果を処理し、エラーと正常結果を判別してログエントリを返す。"""
    # エラー結果の検出
    errors = [r for r in web_results if r.get("source") == "error"]
    if errors:
        err_msg = errors[0].get("title", "不明なエラー")
        entry = f"⚠️ [{label}] 「{query}」→ エラー: {err_msg}"
        return entry, 0

    added = 0
    for r in web_results:
        # 検索候補（searchEntryPoint）は検索結果とは分離して保持する。
        # 結果確認画面での表示専用で、統合LLMへの入力・レポートには含めない
        if r.get("source") == "SearchSuggestions":
            if suggestions_out is not None and r.get("suggestions_html"):
                suggestions_out.append({
                    "query": r.get("query", ""),
                    "html":  r["suggestions_html"],
                })
            continue
        key = r.get("title", "") + r.get("url", "")
        if key not in seen_titles:
            seen_titles.add(key)
            results.append(r)
            added += 1
    entry = f"{icon} [{label}] 「{query}」→ {added}件取得"
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
    # 実行済みクエリ（同一クエリの再実行は無意味なので記録して弾く）。
    # state から引き継ぐことで、再調査ラウンドでも初回に実行済みのクエリを再実行しない
    executed_queries: set[str] = set(state.get("executed_queries", []) or [])
    # 検索候補（Google 検索の searchEntryPoint）。Grounding 結果を表示する際に
    # あわせて表示することが規約で求められるため、結果確認画面用に収集する
    search_suggestions: list[dict] = list(state.get("search_suggestions", []) or [])

    # 進捗のライブ送信（stream_mode="custom"）。invoke 実行時は no-op。
    emit = _make_emit()

    emit("🔎 法令調査を開始します（e-Gov 法令API ＋ Web検索）...")

    search_count = 0  # 実際に検索を実行した回数（スキップは含めない）
    is_reinvestigation = bool(prev_results)

    if is_reinvestigation:
        # 再調査時は前回のシード検索結果を引き継いでいるため、
        # 同じシード検索の再実行はスキップして追加調査（Agenticループ）に進む
        emit("🔁 再調査のため、初回シード検索をスキップして追加調査に進みます...")
    else:
        # シード検索: LLM分析キーワード＋ルールベース必須法令（決定的ベースライン）を
        # 全件検索する。LLMの発想に漏れがあっても、設備属性に対応する必須法令は
        # 必ず検索されることを保証する。
        baseline_keywords = get_suggested_keywords(equipment_info)
        seed_keywords = list(dict.fromkeys(list(initial_keywords) + baseline_keywords))
        emit(
            f"📚 e-Gov 法令API でキーワード検索中..."
            f"（AI分析 {len(initial_keywords)}件＋ルールベース必須法令 {len(baseline_keywords)}件）"
        )
        for kw in seed_keywords:
            laws = search_laws_by_keyword(kw, max_results=4)
            executed_queries.add(kw)
            search_count += 1
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

        def _field_needs_check(key: str) -> bool:
            # 「あり」「不明」「未定」等は条例確認が必要。「なし」明記のみ除外する
            v = str(equipment_info.get(key, "") or "").strip()
            return bool(v) and not v.startswith("なし")

        local_queries = [
            f"横浜市 {equipment_type_str} 届出施設 手続き 規制",
            f"横浜市 {equipment_type_str} 届出 条例 規制",
            f"神奈川県 {equipment_type_str} 届出 条例 規制",
        ]
        # 設備属性に応じて、条例名を明示した検索を追加する
        # （一般語の「条例」検索より市・県の公式手続きページに当たりやすい）
        if _field_needs_check("noise_vibration") or _field_needs_check("wastewater"):
            local_queries.append(
                f"横浜市生活環境の保全等に関する条例 指定事業所 特定施設 届出 {equipment_type_str}"
            )
        if _field_needs_check("chemicals") or _field_needs_check("fire_exhaust"):
            local_queries.append("横浜市火災予防条例 少量危険物 届出 基準")
        for local_query in local_queries:
            web_results = search_web(local_query)
            executed_queries.add(local_query)
            search_count += 1
            entry, added = _process_web_results(web_results, local_query, seen_titles, results, "条例Web", suggestions_out=search_suggestions)
            search_log.append(entry)
            progress_messages.append(AIMessage(content=entry, name="search_progress"))
            emit(entry)

        # 必須Web検索②: 省庁ガイドライン・FAQ
        emit("🌐 省庁ガイドライン・FAQ をWeb検索中...")
        guideline_query = f"{equipment_type_str} 設置 届出 省庁 ガイドライン FAQ"
        web_results = search_web(guideline_query)
        executed_queries.add(guideline_query)
        search_count += 1
        entry, added = _process_web_results(web_results, guideline_query, seen_titles, results, "ガイドラインWeb", suggestions_out=search_suggestions)
        search_log.append(entry)
        progress_messages.append(AIMessage(content=entry, name="search_progress"))
        emit(entry)

        # 必須社内文書検索: 社内規定・社内手続き（登録されている場合のみ）
        if internal_docs_available():
            emit("📁 社内文書（社内規定・過去事例）を検索中...")
            for internal_query in [
                f"{equipment_type_str} 設置 社内手続き 基準",
                "設備導入 社内 安全審査 届出 手続き",
            ]:
                internal_hits = search_internal_docs(internal_query)
                executed_queries.add(internal_query)
                search_count += 1
                entry, added = _process_web_results(internal_hits, internal_query, seen_titles, results, "社内文書", icon="📁")
                search_log.append(entry)
                progress_messages.append(AIMessage(content=entry, name="search_progress"))
                emit(entry)

    # Agenticループ
    # 検索の実行回数（executed_searches）が上限に達するまで回す。
    # 検索済みクエリの再提案はスキップし、検索回数を消費しない。
    # ただしLLM判断回数には別途ハードキャップを設けて無限ループを防ぐ。
    executed_searches = 0
    consecutive_skips = 0
    decision_count = 0
    max_decisions = MAX_SEARCH_ITERATIONS * 2

    while executed_searches < MAX_SEARCH_ITERATIONS and decision_count < max_decisions:
        decision_count += 1
        # 進捗表示：何回目の判断か
        progress_msg = (
            f"🔄 AIが追加調査の要否を判断中"
            f"（検索{executed_searches + 1}件目／最大{MAX_SEARCH_ITERATIONS}件）"
        )
        progress_messages.append(AIMessage(content=progress_msg, name="search_progress"))
        emit(progress_msg)

        issues_str   = "\n".join(f"- {i}" for i in issues)
        keywords_str = "\n".join(f"- {k}" for k in initial_keywords)
        history_str  = "\n".join(search_log) if search_log else "（なし）"
        # タイトルは全件見せて全体像を判断させ、概要は直近10件のみ付けて
        # トークンを抑える（先頭N件だけだと自分の新規取得分が見えなくなる）
        _recent_cutoff = max(0, len(results) - 10)
        results_str  = "\n".join(
            f"- {r.get('title','?')} ({r.get('source','?')})"
            + (
                f"\n    概要: {r.get('snippet','')[:150]}"
                if r.get('snippet') and i >= _recent_cutoff else ""
            )
            for i, r in enumerate(results)
        ) if results else "（なし）"

        note_str = (
            f"\n\n## 🔔 担当者からの追記指示（最優先で考慮し、関連する検索を追加すること）\n{policy_note}"
            if policy_note else ""
        )

        executed_str = "\n".join(f"- {q}" for q in sorted(executed_queries)) if executed_queries else "なし"
        internal_str = (
            "登録あり（search_type=internal で社内規定・過去事例を検索可能）"
            if internal_docs_available()
            else "未登録（search_type=internal は使用禁止）"
        )
        context = (
            f"## 設備情報\n{json.dumps(equipment_info, ensure_ascii=False, indent=2)}\n\n"
            f"## 調査が必要な項目\n{issues_str}\n\n"
            f"## 推奨検索キーワード（参考）\n{keywords_str}\n\n"
            f"## 社内文書インデックス\n{internal_str}\n\n"
            f"## 検索履歴（{decision_count}回目判断）\n{history_str}\n\n"
            f"## ⚠️ 検索済みクエリ（再検索禁止。結果は取得済みで、再実行しても新規情報は得られない）\n{executed_str}\n\n"
            f"## 取得済み法令・情報（{len(results)}件・概要は直近10件のみ表示）\n{results_str}"
            f"{note_str}"
        )

        action: SearchAction = search_llm.invoke([
            SystemMessage(SEARCH_AGENT_SYSTEM),
            HumanMessage(context),
        ])

        if action.done or not action.query:
            done_entry = f"✅ 調査完了（{decision_count}回の判断で収集十分と判断しました）"
            progress_messages.append(AIMessage(content=done_entry, name="search_progress"))
            emit(done_entry)
            break

        # 検索済みクエリの再提案はAPIを呼ばずスキップ（検索回数は消費しない）
        if action.query in executed_queries:
            consecutive_skips += 1
            skip_entry = f"⏭️ スキップ（検索済みクエリの再実行）:「{action.query}」"
            search_log.append(skip_entry)
            progress_messages.append(AIMessage(content=skip_entry, name="search_progress"))
            emit(skip_entry)
            # 新規クエリを提案できなくなったら、これ以上の収集は見込めないため終了
            if consecutive_skips >= 3:
                done_entry = "✅ 調査完了（新規クエリの提案がなくなったため終了します）"
                progress_messages.append(AIMessage(content=done_entry, name="search_progress"))
                emit(done_entry)
                break
            continue

        # 社内文書が未登録なのに internal が選ばれた場合は、検索枠を消費せずスキップ
        if action.search_type == "internal" and not internal_docs_available():
            consecutive_skips += 1
            executed_queries.add(action.query)
            skip_entry = f"⏭️ スキップ（社内文書が未登録のため社内検索は実行不可）:「{action.query}」"
            search_log.append(skip_entry)
            progress_messages.append(AIMessage(content=skip_entry, name="search_progress"))
            emit(skip_entry)
            if consecutive_skips >= 3:
                done_entry = "✅ 調査完了（新規クエリの提案がなくなったため終了します）"
                progress_messages.append(AIMessage(content=done_entry, name="search_progress"))
                emit(done_entry)
                break
            continue

        consecutive_skips = 0
        executed_queries.add(action.query)
        executed_searches += 1

        if action.search_type == "egov":
            emit(f"📚 e-Gov 法令API で追加検索中:「{action.query}」")
            # 1クエリの取得は3件（本法＋施行令＋施行規則で通常足りる。
            # 5件だと周辺法令のノイズが増えて上限を圧迫するため）
            laws = search_laws_by_keyword(action.query, max_results=3)
            search_count += 1
            added = 0
            for law in laws:
                title = law.get("title", "")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    results.append(law)
                    added += 1
            entry = f"🔍 [e-Gov API] 「{action.query}」→ {added}件新規取得"
            search_log.append(entry)
            progress_messages.append(AIMessage(content=entry, name="search_progress"))
            emit(entry)
        elif action.search_type == "internal":
            # 下位の Agentic RAG サブエージェントに委譲する
            # （クエリ展開→ベクトル検索→関連性評価→不足時の再検索を自律実行）
            emit(f"📁 AIエージェントが社内文書を調査中:「{action.query}」")

            def _rag_progress(msg: str) -> None:
                search_log.append(msg)
                progress_messages.append(AIMessage(content=msg, name="search_progress"))
                emit(msg)

            internal_hits = agentic_internal_search(
                action.query,
                context_hint=json.dumps(equipment_info, ensure_ascii=False),
                on_progress=_rag_progress,
            )
            search_count += 1
            entry, added = _process_web_results(internal_hits, action.query, seen_titles, results, "社内文書", icon="📁")
            entry = entry.replace("取得", "新規取得")
            search_log.append(entry)
            progress_messages.append(AIMessage(content=entry, name="search_progress"))
            emit(entry)
        else:
            emit(f"🌐 Web検索中:「{action.query}」")
            web_results = search_web(action.query)
            search_count += 1
            entry, added = _process_web_results(web_results, action.query, seen_titles, results, "AIWeb検索", suggestions_out=search_suggestions)
            entry = entry.replace("取得", "新規取得")
            search_log.append(entry)
            progress_messages.append(AIMessage(content=entry, name="search_progress"))
            emit(entry)

    # ── 網羅性検証: 各論点に対応する情報が集まっているか最終チェック ──
    # Agenticループの done 判断（LLM自己申告）とは独立に、論点×収集結果を
    # 突き合わせ、未カバー論点は補完検索する。それでも残った論点は state に
    # 記録し、結果確認画面・レポートで担当者に明示する。
    uncovered_issues: list[str] = []
    # 論点 → カバー元の法令・情報タイトル（レポート・結果確認でカバー判定の
    # 妥当性を人が検証できるようにする）
    issue_coverage: dict = {}
    if issues:
        emit("🧮 論点ごとの網羅性を検証中...")
        coverage_llm = _llm().with_structured_output(CoverageCheck)

        def _run_coverage_check() -> CoverageCheck | None:
            issues_str  = "\n".join(f"- {i}" for i in issues)
            results_str = "\n".join(
                f"- {r.get('title','?')} ({r.get('source','?')})" for r in results
            ) or "（なし）"
            try:
                return coverage_llm.invoke([
                    SystemMessage(COVERAGE_CHECK_SYSTEM),
                    HumanMessage(
                        f"## 調査が必要な論点\n{issues_str}\n\n"
                        f"## 収集済みの法令・情報一覧\n{results_str}"
                    ),
                ])
            except Exception:
                return None

        check = _run_coverage_check()
        uncovered = check.uncovered if check else []
        if uncovered:
            emit(f"⚠️ 未カバーの論点が{len(uncovered)}件。全件を補完検索します...")
            _fill_start = len(results)
            for u in uncovered:
                q = (u.suggested_query or "").strip()
                if not q or q in executed_queries:
                    continue
                executed_queries.add(q)
                search_count += 1
                if u.search_type == "egov":
                    emit(f"📚 [網羅性補完] e-Gov検索:「{q}」")
                    laws = search_laws_by_keyword(q, max_results=5)
                    added = 0
                    for law in laws:
                        title = law.get("title", "")
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            results.append(law)
                            added += 1
                    entry = f"🔍 [網羅性補完] 「{q}」→ {added}件取得"
                elif u.search_type == "internal":
                    emit(f"📁 [網羅性補完] AIエージェントが社内文書を調査:「{q}」")
                    internal_hits = agentic_internal_search(
                        q,
                        context_hint=json.dumps(equipment_info, ensure_ascii=False),
                        on_progress=emit,
                    )
                    entry, added = _process_web_results(internal_hits, q, seen_titles, results, "網羅性補完(社内)", icon="📁")
                else:
                    emit(f"🌐 [網羅性補完] Web検索:「{q}」")
                    web_results = search_web(q)
                    entry, added = _process_web_results(web_results, q, seen_titles, results, "網羅性補完", suggestions_out=search_suggestions)
                search_log.append(entry)
                progress_messages.append(AIMessage(content=entry, name="search_progress"))
                emit(entry)

            # 補完検索で取得した結果は、上限超過時の割愛対象から保護する
            # （網羅性チェックを通すために取得した根拠が、統合前に割愛で
            # 消えて「チェックOKなのに根拠なし」になる矛盾を防ぐ）
            for r in results[_fill_start:]:
                r["protected"] = True

            # 補完後に再チェックし、それでも未カバーの論点を記録する
            recheck = _run_coverage_check()
            check = recheck or check
            uncovered_issues = [u.issue for u in recheck.uncovered] if recheck else []
            if uncovered_issues:
                warn = (
                    f"⚠️ 補完検索後も{len(uncovered_issues)}件の論点は対応情報が"
                    f"見つかりませんでした。結果確認画面・レポートに明示します。"
                )
                progress_messages.append(AIMessage(content=warn, name="search_progress"))
                emit(warn)
            else:
                emit("✅ 補完検索により全論点がカバーされました")
        else:
            emit("✅ 全論点に対応する情報が収集されています")

        # 最終チェック結果からカバー元マップを構築（入力論点の原文表記のみ採用）
        if check:
            _issue_set = set(issues)
            issue_coverage = {
                c.issue: [t for t in c.covered_by if t][:3]
                for c in check.covered
                if c.issue in _issue_set and c.covered_by
            }

    # ── 上限超過時の切り捨て ──
    # ソース別のバランスを保って割愛する。Web結果は条例・届出先・手続き情報の
    # 主要ソースのため最低保持枠を確保し、e-Gov は必須法令・分析キーワード由来を
    # 優先して残す（e-Gov だけで上限を埋めて Web が全滅すると、統合LLMが
    # 届出先を判断できなくなる）。社内文書は全件保持する。
    WEB_MIN_KEEP = 15
    if len(results) > MAX_SEARCH_RESULTS:
        # 網羅性補完で取得した結果（protected）は無条件に保持し、
        # 残りの枠にソース別クォータを適用する
        protected_res = [r for r in results if r.get("protected")]
        rest          = [r for r in results if not r.get("protected")]
        budget = max(0, MAX_SEARCH_RESULTS - len(protected_res))

        def _src(r: dict) -> str:
            s = r.get("source", "")
            if s == "社内文書":
                return "internal"
            return "egov" if s.startswith("e-Gov") else "web"
        internal_res = [r for r in rest if _src(r) == "internal"]
        egov_res     = [r for r in rest if _src(r) == "egov"]
        web_res      = [r for r in rest if _src(r) == "web"]

        web_quota = min(
            len(web_res),
            max(WEB_MIN_KEEP, budget - len(internal_res) - len(egov_res)),
        )
        egov_quota = max(0, budget - len(internal_res) - web_quota)

        dropped_egov = max(0, len(egov_res) - egov_quota)
        dropped_web  = max(0, len(web_res) - web_quota)
        if dropped_egov:
            # ルールベース必須法令・AI分析キーワード由来の法令を優先して残す
            prio = set(initial_keywords) | set(get_suggested_keywords(equipment_info))
            egov_res = sorted(
                egov_res, key=lambda r: 0 if r.get("keyword", "") in prio else 1
            )[:egov_quota]
        results = (
            protected_res + internal_res + egov_res + web_res[:web_quota]
        )[:MAX_SEARCH_RESULTS]
        warn = (
            f"⚠️ 収集件数が上限{MAX_SEARCH_RESULTS}件を超えたため、"
            f"e-Gov {dropped_egov}件・Web {dropped_web}件を割愛しました"
            f"（必須法令・社内文書・網羅性補完の取得分・Web最低{WEB_MIN_KEEP}件枠を優先保持）"
        )
        progress_messages.append(AIMessage(content=warn, name="search_progress"))
        emit(warn)

    egov_count     = len([r for r in results if "e-Gov" in r.get("source", "")])
    internal_count = len([r for r in results if r.get("source", "") == "社内文書"])
    web_count      = len(results) - egov_count - internal_count
    summary_msg = (
        f"調査完了（{search_count}回検索）。"
        f"e-Gov {egov_count}件、Web {web_count}件"
        + (f"、社内文書 {internal_count}件" if internal_count else "")
        + f"、計{len(results)}件を収集しました。"
    )
    progress_messages.append(AIMessage(content=summary_msg))
    emit(summary_msg)
    emit("🧩 調査結果を統合し、法令別の対応事項を整理中...")

    return {
        "search_results":   results[:MAX_SEARCH_RESULTS],
        "executed_queries": sorted(executed_queries),
        "uncovered_issues": uncovered_issues,
        "issue_coverage":   issue_coverage,
        "search_suggestions": search_suggestions,
        "phase":            "synthesizing",
        "messages":         progress_messages,
    }


# ─── 条番号グラウンディング ──────────────────────────────────────
def _ground_relevant_articles(law_items: list) -> None:
    """relevant_articles / deliveries.law_article をe-Gov原文に基づいて確定する（in-place）。

    LLMの知識由来の条番号はハルシネーションを含むため使用せず、
    e-Govから取得した実在の条一覧（条番号＋見出し）の中からLLMに選択させ、
    一覧に実在しない条番号はすべて除去する。
    e-Govで原文を取得できない法令（条例等）は根拠を検証できないため条番号を表示しない。
    """
    emit = _make_emit()

    selector_llm = _llm().with_structured_output(ArticleSelection)

    for law in law_items:
        law_name = law.get("law_name", "")
        article_list = fetch_article_list(
            law.get("law_id", ""), law.get("law_revision_id", "")
        )

        if not article_list:
            # 原文未取得の法令は未検証の条番号を残さない（表示側は「条番号確認中」になる）
            law["relevant_articles"] = []
            for d in law.get("deliveries", []):
                d["law_article"] = ""
            continue

        emit(f"📖 「{law_name}」の条番号を e-Gov 原文と照合中...")
        valid_refs = {a["ref"] for a in article_list}
        listing = "\n".join(f'- {a["ref"]}{a["caption"]}' for a in article_list)
        deliveries_str = "\n".join(
            f"- {d.get('item','')}" for d in law.get("deliveries", []) if d.get("item")
        ) or "（なし）"
        ctx = (
            f"## 法令名\n{law_name}\n\n"
            f"## この設備への適用理由\n{law.get('applicability', '')}\n\n"
            f"## 想定される届出・申請\n{deliveries_str}\n\n"
            f"## 条文一覧（この中からのみ選択すること）\n{listing}"
        )
        try:
            sel: ArticleSelection = selector_llm.invoke([
                SystemMessage(ARTICLE_SELECTION_SYSTEM),
                HumanMessage(ctx),
            ])
            picked = [normalize_article_ref(a) for a in sel.articles]
            law["relevant_articles"] = [
                p for p in dict.fromkeys(picked) if p in valid_refs
            ][:5]
        except Exception:
            law["relevant_articles"] = []

        # deliveries.law_article も実在検証（「第◯条第◯項」は条部分で照合）
        for d in law.get("deliveries", []):
            ref = normalize_article_ref(d.get("law_article", ""))
            base = re.sub(r"第\d+項$", "", ref)
            d["law_article"] = ref if base and base in valid_refs else ""


# ─── 統合前の条文プリフェッチ ─────────────────────────────────────
MAX_PREFETCH_LAWS = 8
MAX_PREFETCH_ARTICLES = 3


def _prefetch_article_excerpts(
    equipment_info: dict, issues: list, search_results: list,
    priority_keywords: set, emit,
) -> str:
    """統合前に主要候補法令の関連条文を e-Gov 原文から取得する。

    e-Gov 検索結果は法令名しか持たないため、そのままでは適用判断が LLM の
    内部知識頼みになる。ここで候補法令ごとに「適用範囲・届出義務・数量閾値」
    に関わる条文を取得し、統合 LLM の判断根拠（特に数量閾値の照合）に使わせる。
    """
    egov_laws = [
        r for r in search_results
        if r.get("source", "").startswith("e-Gov") and r.get("law_id")
    ]
    if not egov_laws:
        return ""

    # ルールベース必須法令・分析キーワードでヒットした法令を優先する
    prioritized = sorted(
        egov_laws,
        key=lambda r: 0 if r.get("keyword", "") in priority_keywords else 1,
    )
    candidates: list[dict] = []
    seen_ids: set[str] = set()
    for r in prioritized:
        if r["law_id"] in seen_ids:
            continue
        seen_ids.add(r["law_id"])
        candidates.append(r)
        if len(candidates) >= MAX_PREFETCH_LAWS:
            break

    selector_llm = _llm().with_structured_output(ArticleSelection)
    info_str = json.dumps(equipment_info, ensure_ascii=False, indent=2)
    issues_str = "\n".join(f"- {i}" for i in issues)

    blocks: list[str] = []
    for law in candidates:
        title = law.get("title", "")
        article_list = fetch_article_list(
            law["law_id"], law.get("law_revision_id", "")
        )
        if not article_list:
            continue  # 条例など原文未収載の法令はスキップ
        emit(f"📖 条文照合: 「{title}」の適用範囲・閾値条文を取得中...")
        listing = "\n".join(f'- {a["ref"]}{a["caption"]}' for a in article_list)
        try:
            sel: ArticleSelection = selector_llm.invoke([
                SystemMessage(PRE_SYNTHESIS_ARTICLE_SYSTEM),
                HumanMessage(
                    f"## 法令名\n{title}\n\n"
                    f"## 設備情報\n{info_str}\n\n"
                    f"## 調査論点\n{issues_str}\n\n"
                    f"## 条文一覧（この中からのみ選択すること）\n{listing}"
                ),
            ])
        except Exception:
            continue

        valid_refs = {a["ref"] for a in article_list}
        refs = [
            r for r in dict.fromkeys(normalize_article_ref(a) for a in sel.articles)
            if r in valid_refs
        ][:MAX_PREFETCH_ARTICLES]
        if not refs:
            continue
        try:
            texts = fetch_article_text(law["law_id"], refs, law.get("law_revision_id", ""))
        except Exception:
            texts = {}
        lines = [f"### {title}"]
        for ref in refs:
            t = (texts.get(ref) or "").strip()
            if t:
                lines.append(f"◆ {ref}\n{t[:600]}")
        if len(lines) > 1:
            blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


# ─── 届出先出典の解決 ─────────────────────────────────────────────
def _attach_delivery_sources(law_items: list, search_results: list) -> None:
    """deliveries.authority_source_id（[R◯]）を検索結果のURL・タイトルに解決して付与する（in-place）。
    LLMにURLを直接書かせるとハルシネーションするため、IDで参照させて
    実在する検索結果から決定的に解決する。実在しないIDはリンクなしになるだけ。"""
    by_id = {
        f"R{i}": r for i, r in enumerate(search_results[:MAX_SEARCH_RESULTS], 1)
    }
    for law in law_items:
        for d in law.get("deliveries", []):
            sid = str(d.get("authority_source_id", "") or "").strip().strip("[]")
            r = by_id.get(sid)
            if r and r.get("url"):
                d["authority_source_url"]   = r["url"]
                d["authority_source_title"] = r.get("title", "")
                # レポート生成時に Grounding 由来リンクを除外するための種別
                # （Gemini API 追加利用規約: Grounding 結果のリンクは保存不可）
                d["authority_source_kind"]  = r.get("source", "")
            else:
                d["authority_source_url"]   = ""
                d["authority_source_title"] = ""
                d["authority_source_kind"]  = ""


# ─── 法令名マッチング ─────────────────────────────────────────────
# 本法と下位法令の取り違えを防ぐための接尾辞
_SUBLAW_SUFFIXES = ("施行令", "施行規則", "施行細則")


def _is_law_name_match(law_name: str, title: str) -> bool:
    """法令名と e-Gov 検索結果タイトルが同一法令を指すか判定する。

    完全一致のほか部分一致も許容するが、一方が他方＋「施行令」「施行規則」等の
    下位法令名になっている組み合わせは別法令として除外する
    （例：law_name「消防法」と title「消防法施行規則」は不一致）。
    """
    if not law_name or not title:
        return False
    if law_name == title:
        return True
    for a, b in ((law_name, title), (title, law_name)):
        if b.startswith(a) and b[len(a):].startswith(_SUBLAW_SUFFIXES):
            return False
    return law_name in title or title in law_name


# ─── ノード: 結果統合 ─────────────────────────────────────────────
def synthesis_node(state: AppState) -> dict:
    """検索結果を統合してアクションアイテムを生成し、state に保存する。
    結果レビュー（interrupt）は後続の results_review_node で行う。
    interrupt を同一ノード内に置くと resume 時にノード全体（統合LLM＋
    条番号グラウンディング）が再実行され、確認した law_items とレポートに
    載る law_items がズレる・承認後の待ち時間とコストが倍になるため分離。"""
    structured_llm = _llm().with_structured_output(SynthesisResult)
    emit = _make_emit()

    equipment_info = state.get("equipment_info", {})
    search_results = state.get("search_results", [])
    issues         = state.get("issues", [])
    policy_note    = state.get("policy_note", "")

    # 統合前に主要候補法令の条文を取得し、適用判断・数量閾値照合の根拠にする
    # （e-Gov検索結果は法令名のみで、これが無いと判断がLLMの記憶頼みになる）。
    # 再調査後の再統合で法令候補が変わっていなければ前回の抜粋を再利用し、
    # 条文選択LLM呼び出し（最大8回）を省く
    egov_law_ids = sorted({
        r.get("law_id") for r in search_results
        if r.get("source", "").startswith("e-Gov") and r.get("law_id")
    })
    cache = state.get("prefetch_cache") or {}
    if cache.get("law_ids") == egov_law_ids and cache.get("excerpts"):
        article_excerpts = cache["excerpts"]
        emit("📖 条文照合: 前回取得済みの条文抜粋を再利用します")
    else:
        priority_keywords = (
            set(state.get("search_keywords", []) or [])
            | set(get_suggested_keywords(equipment_info))
        )
        try:
            article_excerpts = _prefetch_article_excerpts(
                equipment_info, issues, search_results, priority_keywords, emit,
            )
        except Exception:
            article_excerpts = ""

    emit("🧩 条文抜粋と調査結果をもとに、法令別の対応事項を統合中...")
    context = (
        f"【設備情報】\n{json.dumps(equipment_info, ensure_ascii=False, indent=2)}\n\n"
        f"【確認が必要な論点】\n{json.dumps(issues, ensure_ascii=False)}\n\n"
        + (f"【担当者からの追記指示（最優先で反映）】\n{policy_note}\n\n" if policy_note else "")
        + (f"【主要候補法令の条文抜粋（e-Gov原文）】\n{article_excerpts}\n\n" if article_excerpts else "")
        + f"【e-Gov・Web・社内文書 調査結果（抜粋・[R◯]は出典ID）】\n"
        + "\n".join(
            f"- [R{i}] {r.get('title','?')} ({r.get('source','?')})"
            + (f"\n  概要: {r.get('snippet','')[:400]}" if r.get('snippet') else "")
            for i, r in enumerate(search_results[:MAX_SEARCH_RESULTS], 1)
        )
    )

    result: SynthesisResult = structured_llm.invoke([
        SystemMessage(SYNTHESIS_SYSTEM),
        HumanMessage(context),
    ])

    law_items = [item.model_dump() for item in result.law_items]

    # 「情報不足で非該当と断定できない」項目は非該当リストに置かず要確認（check）へ移動する。
    # LLM が「チェックリストは law_items か excluded_laws のどちらかに必ず入れる」ルールに
    # 引きずられ、判定不能の法令を excluded_laws に押し込む誤りが実際に起きたためのガード。
    _INSUFFICIENT_KEYWORDS = ("不明", "断定できない", "判定不能", "判断できない", "情報不足", "未記入")
    excluded_laws: list[dict] = []
    existing_names = {item.get("law_name", "") for item in law_items}
    for e in result.excluded_laws:
        insufficient = (
            e.basis == "insufficient_info"
            or any(k in e.reason for k in _INSUFFICIENT_KEYWORDS)
        )
        if not insufficient:
            excluded_laws.append({"law_name": e.law_name, "reason": e.reason})
        elif e.law_name not in existing_names:
            law_items.append({
                "law_name":         e.law_name,
                "applicability": (
                    "AIは非該当の可能性が高いと判断しましたが、判断材料が不足して"
                    "いるため断定せず「要確認」としています。設備情報を確認のうえ"
                    f"適用要否を判定してください。（AIの判断内容：{e.reason}）"
                ),
                "priority":         "check",
                "relevant_articles": [],
                "deliveries":       [],
                "internal_actions": [],
            })
            existing_names.add(e.law_name)

    # ── 確認漏れ防止の自己修復 ──
    # (1) ルールベース必須法令との突合で最終リストから欠落した法令、
    # (2) 統合LLMが挙げた「調査結果に根拠がない追加調査候補（additional_check_laws）」を、
    # e-Gov実在確認＋LLM追加判定で「要確認」か「非該当＋理由」に必ず振り分ける。
    # （従来はレポートの突合チェックで「⚠️ 未掲載」警告を出すだけだった）
    search_results = state.get("search_results", [])
    try:
        def _hit(base: str, names) -> bool:
            cands = {c for c in (base, resolve_law_alias(base)) if c}
            return any(n and any(c in n or n in c for c in cands) for n in names)

        current_names = (
            [l.get("law_name", "") for l in law_items]
            + [e.get("law_name", "") for e in excluded_laws]
        )
        candidates: list[dict] = []
        for base in get_suggested_keywords(equipment_info):
            if not _hit(base, current_names):
                candidates.append({
                    "law_name": resolve_law_alias(base),
                    "origin":   "設備属性から機械的に導出した必須確認法令",
                    "reason":   "設備属性に対応する必須確認法令だが、調査の最終リストに含まれていない",
                })
        for a in result.additional_check_laws:
            name = resolve_law_alias(str(a.law_name or "").strip())
            if name and not _hit(name, current_names) \
                    and not _hit(name, [c["law_name"] for c in candidates]):
                candidates.append({
                    "law_name": name,
                    "origin":   "統合AIが一般知識から挙げた追加調査候補",
                    "reason":   a.reason,
                })
        candidates = candidates[:10]

        if candidates:
            emit(
                f"🛡️ 確認漏れ防止チェック: 最終リストにない候補 {len(candidates)}件を追加判定中..."
            )
            # e-Gov で実在確認と law_id 取得（ヒットは情報源リストにも加える）
            seen_titles = {r.get("title", "") for r in search_results}
            for c in candidates:
                try:
                    hits = search_laws_by_keyword(c["law_name"], max_results=3)
                except Exception:
                    hits = []
                hit = next(
                    (h for h in hits if _is_law_name_match(c["law_name"], h.get("title", ""))),
                    None,
                )
                if hit:
                    c["law_id"]          = hit.get("law_id", "")
                    c["law_revision_id"] = hit.get("law_revision_id", "")
                    if hit.get("title") and hit["title"] not in seen_titles:
                        seen_titles.add(hit["title"])
                        search_results.append(hit)

            supp_llm = _llm().with_structured_output(SupplementalResult)
            cand_str = "\n".join(
                f"- {c['law_name']}（出所: {c['origin']}）\n  経緯・理由: {c['reason']}"
                for c in candidates
            )
            supp: SupplementalResult = supp_llm.invoke([
                SystemMessage(SUPPLEMENTAL_LAW_SYSTEM),
                HumanMessage(
                    f"## 設備情報\n{json.dumps(equipment_info, ensure_ascii=False, indent=2)}\n\n"
                    f"## 追加判定候補の法令（全件判定すること）\n{cand_str}"
                ),
            ])
            judgments = list(supp.judgments)

            n_check = n_excl = 0
            for c in candidates:
                j = next(
                    (v for v in judgments if _is_law_name_match(c["law_name"], v.law_name)),
                    None,
                )
                # 判定が返らなかった候補は安全側（要確認）に倒す
                verdict = j.verdict if j else "check"
                reason  = j.reason if j else "自動判定を取得できなかったため、安全側で要確認としています"
                if verdict == "excluded":
                    n_excl += 1
                    excluded_laws.append({
                        "law_name": c["law_name"],
                        "reason":   f"（確認漏れ防止チェックでの追加判定）{reason}",
                    })
                    continue
                n_check += 1
                law_items.append({
                    "law_name": c["law_name"],
                    "applicability": (
                        f"🛡️ 確認漏れ防止チェックによる自動追加（{c['origin']}）。{reason}"
                    ),
                    "priority":          "check",
                    "relevant_articles": [],
                    "deliveries":        [],
                    "internal_actions":  [{
                        "item":        "本法令の適用要否の確認（仕様・数量等の確定後に判定）",
                        "responsible": "設備導入担当者",
                        "deadline":    "稼働前",
                    }],
                    "law_id":            c.get("law_id", ""),
                    "law_revision_id":   c.get("law_revision_id", ""),
                })
            emit(
                f"🛡️ 確認漏れ防止チェック完了: 要確認 {n_check}件を追加・"
                f"非該当 {n_excl}件を判定しました"
            )
    except Exception:
        # 自己修復の失敗で調査全体を落とさない（レポート側の突合チェックが最後の砦になる）
        emit("⚠️ 確認漏れ防止チェックの自動判定に失敗しました（レポートの突合チェックで確認してください）")

    # e-Gov の law_id / law_revision_id を法令名で逆引きして付与
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
        # 完全一致を最優先し、部分一致は候補の中から名称が最も近いものを選ぶ
        # （単純な部分一致だと「消防法」が検索結果の並び順次第で「消防法施行規則」に
        # 先にマッチし、本法の項目に施行規則の条文が表示される取り違えが起きたため）
        ids = title_to_ids.get(law_name)
        if ids is None:
            candidates = [
                (title, i) for title, i in title_to_ids.items()
                if _is_law_name_match(law_name, title)
            ]
            if candidates:
                _, ids = min(candidates, key=lambda c: abs(len(c[0]) - len(law_name)))
        if ids:
            law["law_id"]          = ids["law_id"]
            law["law_revision_id"] = ids["law_revision_id"]
        else:
            law.setdefault("law_id", "")
            law.setdefault("law_revision_id", "")

    # law_id が未解決の法令だけ e-Gov API で補完検索
    # （名称が一致するヒットのみ採用する。無条件に1件目を採用すると
    # 別法令の law_id が付き、無関係な条文が表示されるため）
    for law in law_items:
        if not law.get("law_id"):
            law_name = law.get("law_name", "")
            if law_name:
                hits = search_laws_by_keyword(law_name, max_results=3)
                hit = next(
                    (h for h in hits if _is_law_name_match(law_name, h.get("title", ""))),
                    None,
                )
                if hit:
                    law["law_id"]          = hit.get("law_id", "")
                    law["law_revision_id"] = hit.get("law_revision_id", "")

    # 条番号をe-Gov原文と照合して確定（LLM知識由来の条番号ハルシネーション防止）
    _ground_relevant_articles(law_items)

    # 届出先の出典ID（[R◯]）を実在する検索結果のURL・タイトルに解決
    _attach_delivery_sources(law_items, search_results)

    # 優先度別件数は、要確認への昇格・確認漏れ防止チェックの追加分を含む
    # 最終的な law_items から再集計する（LLM出力の件数は加工前の値のためズレる）
    risk_count = {
        p: sum(1 for l in law_items if l.get("priority", "check") == p)
        for p in ("required", "check", "pending")
    }

    return {
        "law_items":         law_items,
        "synthesis_summary": result.summary,
        "risk_count":        risk_count,
        # LLM生出力ではなく、情報不足の非該当を「要確認」へ昇格させた後の
        # 選別済みリストを返す（生出力を返すと要確認と非該当に二重掲載される）
        "excluded_laws":     excluded_laws,
        # 確認漏れ防止チェックで追加取得した e-Gov ヒットを含む
        "search_results":    search_results,
        "prefetch_cache":    {"law_ids": egov_law_ids, "excerpts": article_excerpts},
        "phase":             "results_review",
    }


# ─── ノード: 結果レビュー（Human in the loop） ────────────────────
def results_review_node(state: AppState) -> dict:
    """調査結果のレビュー。ノード先頭で interrupt するため、再開時に
    統合LLM・条番号グラウンディングは再実行されず、確認した law_items が
    そのままレポートに使われる。"""
    decision = interrupt({
        "phase":            "results_review",
        "law_items":        state.get("law_items", []),
        "summary":          state.get("synthesis_summary", ""),
        "risk_count":       state.get("risk_count", {}),
        "excluded_laws":    state.get("excluded_laws", []),
        "uncovered_issues": state.get("uncovered_issues", []),
        "issues":           state.get("issues", []),   # 網羅性チェック表示用
        "issue_coverage":   state.get("issue_coverage", {}),  # カバー元の法令表示用
    })

    # 「足りない・再調査して」依頼なら検索フェーズに戻す
    if _is_reinvestigate(decision):
        extra = _extract_after(decision, _REINVEST_PREFIXES)
        req_text, has_docs = _split_attachments(extra)
        label = req_text or "添付資料の内容を踏まえた再調査"
        if has_docs:
            label += "（添付資料の内容は調査の参考情報として反映）"
        return {
            "messages":    [AIMessage(f"担当者から追加調査の依頼を受けました：{label}\n再調査を実施します...")],
            "policy_note": _merge_note(state.get("policy_note", ""), extra),
            "issues":      list(state.get("issues", []) or []) + [f"【追加調査依頼】{req_text or label}"],
            "phase":       "searching",
        }

    # 通常: レポートへ。case_id をここで確定し、以降の再実行でも不変にする
    case_id = state.get("case_id") or (
        f"EQ-{datetime.datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
    )
    return {
        "messages": [AIMessage("レビュー完了。レポートを生成します。")],
        "case_id":  case_id,
        "phase":    "reporting",
    }


# ─── 文面修正（LLM）ヘルパー ──────────────────────────────────────
def _refine_law_items(law_items: list, equipment_info: dict, instruction: str) -> list:
    """担当者の文面修正依頼に沿って law_items のテキストを LLM で調整する。
    法令名・条番号・届出先などの事実情報は依頼が無い限り変えない。
    失敗時は元の law_items を返す。"""
    if not law_items:
        return law_items
    refine_llm = _llm().with_structured_output(RefineResult)
    ctx = (
        f"【設備情報】\n{json.dumps(equipment_info, ensure_ascii=False, indent=2)}\n\n"
        f"【現在の法令別対応事項(JSON)】\n{json.dumps(law_items, ensure_ascii=False, indent=2)}\n\n"
        f"【担当者からの文面修正依頼】\n{instruction}\n\n"
        "上記の依頼に沿って law_items の文面（applicability / item / deadline / responsible など）"
        "を読みやすく修正してください。法令名・条番号(relevant_articles, law_article)・"
        "届出先(authority)・届出先の根拠(authority_basis)・priority などの事実情報は、"
        "依頼で明示されない限り変更しないこと。"
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
        excluded_laws=state.get("excluded_laws", []),
        uncovered_issues=state.get("uncovered_issues", []),
        summary=state.get("synthesis_summary", ""),
        issues=state.get("issues", []),
        issue_coverage=state.get("issue_coverage", {}),
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
        req_text, has_docs = _split_attachments(extra)
        label = req_text or "添付資料の内容を踏まえた再調査"
        if has_docs:
            label += "（添付資料の内容は調査の参考情報として反映）"
        return {
            "messages":    [AIMessage(f"レポート確認後、追加調査の依頼を受けました：{label}\n再調査を実施します...")],
            "policy_note": _merge_note(state.get("policy_note", ""), extra),
            "issues":      list(state.get("issues", []) or []) + [f"【追加調査依頼】{req_text or label}"],
            "case_id":     case_id,
            "phase":       "searching",
        }

    # ② 文面修正（LLM）→ law_items を更新し report を再実行して再確認させる
    refine = _extract_after(decision, _REFINE_PREFIXES)
    if refine:
        new_items = _refine_law_items(
            state.get("law_items", []), state.get("equipment_info", {}), refine
        )
        # スキーマ往復で消えた出典URL・タイトルを authority_source_id から再解決
        _attach_delivery_sources(new_items, state.get("search_results", []))
        refine_label, refine_docs = _split_attachments(refine)
        if refine_docs:
            refine_label += "（添付資料の内容を参照）"
        return {
            "messages":  [AIMessage(f"レポート文面の修正依頼を受けました：{refine_label}\n修正して再生成します...")],
            "law_items": new_items,
            "case_id":   case_id,
            "phase":     "reporting",
        }

    # ③ 承認 → 完了。承認済みの結論をケースメモリに保存する（CBR: Retain）
    memory_note = ""
    try:
        saved = save_case(
            case_id=case_id,
            equipment_info=state.get("equipment_info", {}),
            law_items=state.get("law_items", []),
            excluded_laws=state.get("excluded_laws", []),
        )
        if saved:
            memory_note = (
                "\n\n📚 この案件は「過去の調査事例」に保存されました。"
                "次回以降、類似設備の案件でAIが自動で参考にします。"
            )
    except Exception:
        # 保存失敗で承認フローを止めない
        pass

    return {
        "messages": [AIMessage(
            "✅ レポートが承認されました。\n\n"
            "設備導入時の法令・手続き確認が完了しました。\n"
            "レポートをダウンロードして、担当部署に共有してください。"
            + memory_note
        )],
        "report_html": report_html,
        "case_id":     case_id,
        "phase":       "complete",
    }


# ─── グラフ構築 ───────────────────────────────────────────────────
def _route_start(state: AppState) -> str:
    return {
        "hearing":        "hearing",
        "analysis":       "analysis",
        "policy_review":  "policy_review",
        "searching":      "search",
        "synthesizing":   "synthesis",
        "results_review": "results_review",
        "reporting":      "report",
    }.get(state.get("phase", "hearing"), "hearing")


def _route_after_results_review(state: AppState) -> str:
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

    builder.add_node("hearing",        hearing_node)
    builder.add_node("analysis",       analysis_node)
    builder.add_node("policy_review",  policy_review_node)
    builder.add_node("search",         search_node)
    builder.add_node("synthesis",      synthesis_node)
    builder.add_node("results_review", results_review_node)
    builder.add_node("report",         report_node)

    builder.add_conditional_edges(START, _route_start, {
        "hearing":        "hearing",
        "analysis":       "analysis",
        "policy_review":  "policy_review",
        "search":         "search",
        "synthesis":      "synthesis",
        "results_review": "results_review",
        "report":         "report",
    })
    builder.add_conditional_edges("hearing", _route_hearing, {
        "analysis": "analysis",
        END: END,
    })
    builder.add_edge("analysis",      "policy_review")
    builder.add_edge("policy_review", "search")
    builder.add_edge("search",        "synthesis")
    builder.add_edge("synthesis",     "results_review")
    # 結果確認で「再調査」依頼 → search に戻す。それ以外は report へ
    builder.add_conditional_edges("results_review", _route_after_results_review, {
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
