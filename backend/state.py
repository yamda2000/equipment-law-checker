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

    # 方針確認で担当者が追記した指示（調査・統合に反映）
    policy_note: str

    # 検索結果
    search_results: list

    # 法令別アクションアイテム
    law_items: list  # LawItem の dict リスト

    # レポート
    report_html: str
    case_id: str

    # エラー
    error: Optional[str]
