"""LangGraph 状態定義"""

from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AppState(TypedDict):
    # 会話履歴（add_messages で自動マージ）
    messages: Annotated[list, add_messages]

    # ワークフローフェーズ
    phase: str  # hearing | analysis | searching | synthesizing | reporting | complete

    # ヒアリング結果
    equipment_info: dict
    hearing_complete: bool

    # 分析結果
    issues: list
    unknown_items: list
    search_keywords: list
    search_plan: str
    analysis_summary: str

    # 方針確認で担当者が追記した指示（調査・統合に反映）
    policy_note: str

    # 検索結果
    search_results: list
    # 実行済み検索クエリ（再調査ラウンドで同一クエリの再実行を防ぐ）
    executed_queries: list
    # 網羅性検証で「対応情報が見つからない」と判定された論点（担当者に明示）
    uncovered_issues: list

    # 統合前にe-Govから取得した条文抜粋のキャッシュ
    # {"law_ids": [...], "excerpts": str}。再調査後の再統合で法令候補が
    # 変わっていなければ再利用し、条文選択LLM呼び出し（最大8回）を省く
    prefetch_cache: dict

    # 法令別アクションアイテム
    law_items: list  # LawItem の dict リスト
    synthesis_summary: str
    risk_count: dict
    # 確認したが非該当と判断した法令（理由つき・確認漏れ防止のため明示）
    excluded_laws: list

    # レポート
    report_html: str
    case_id: str

    # エラー
    error: Optional[str]
