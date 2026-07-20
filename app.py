"""
設備導入時 法令・届出施設確認サポートAI
メインアプリ - Streamlit UI
"""

import os
import re
import html as html_lib
import hashlib
import time
import uuid
import sys
import logging
import traceback
import urllib.parse
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.tracers.context import register_configure_hook
from langgraph.types import Command

from backend.workflow import (
    workflow, get_interrupt_data, get_all_messages, get_state_value,
    HEARING_COMPLETE_MARKER,
)
from backend.tools.egov import (
    fetch_article_text, fetch_article_captions, fetch_article_chapters,
    article_sort_key,
)
from backend.doc_intake import extract_text_from_file, extract_equipment_info
from backend.tools.internal_docs import (
    list_registered as list_internal_docs,
    ingest_files as ingest_internal_docs,
    delete_document as delete_internal_doc,
    delete_all as delete_all_internal_docs,
)
from backend.case_memory import list_cases, delete_case
from backend.report_gen import ordinance_links_html
from backend.tools.web_search import collect_gemini_usage, search_web
from backend.observability import trace_run, langfuse_enabled, get_session_cost
from backend.qa import (
    answer_question, generate_document,
    build_context as build_qa_context, build_history as build_qa_history,
)
from backend.doc_export import build_file as build_doc_file

# ─────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────
st.set_page_config(
    page_title="法令・届出施設確認サポートAI",
    page_icon="⚖️",
    layout="wide",
)

st.markdown("""
<style>
/* アプリを80%サイズで表示する。
   body に zoom をかけると Streamlit の 100vh 基準の高さ計算とズレて
   画面下部が描画されなくなるため、スクロール領域内のコンテンツ
   コンテナ（メイン・サイドバー）にのみ適用する */
.block-container { zoom: 0.8; }
section[data-testid="stSidebar"] > div { zoom: 0.8; }

.stApp { background-color: #f5f7fa; }

.ai-bubble {
    background: #ffffff;
    border-left: 4px solid #1565C0;
    border-radius: 0 12px 12px 12px;
    padding: 14px 18px;
    margin: 6px 0 2px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
}
.ai-header { color: #1565C0; font-size: 14px; font-weight: 700; margin-bottom: 6px; }

.user-bubble {
    background: #E3F2FD;
    border-right: 4px solid #42A5F5;
    border-radius: 12px 0 12px 12px;
    padding: 12px 16px;
    margin: 6px 0 2px 0;
    margin-left: 15%;
}
.user-header { color: #1976D2; font-size: 14px; font-weight: 700; margin-bottom: 4px; text-align: right; }
.user-body { text-align: right; }

.system-bubble {
    background: #E3F2FD;
    border-left: 4px solid #1565C0;
    border-radius: 0 12px 12px 12px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #0D47A1;
}

/* 右上固定の「AIに質問」フローティングパネル（全フェーズ共通のQA）。
   決定ボタン（案件を進める操作）と物理的に離し、どの画面でも同じ位置に出す。
   top はズーム(0.8)適用後にタイトル行の高さ（Streamlitヘッダー直下）に
   揃うよう 90px（実効72px）とする */
.st-key-qa_float {
    position: fixed;
    top: 90px;
    right: 32px;
    width: auto;
    z-index: 1000;
}
.st-key-qa_float [data-testid="stPopover"] button {
    background: #5E35B1;
    color: white;
    border: none;
    border-radius: 32px;
    padding: 16px 34px;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(0,0,0,.28);
}
/* ボタン文言は内側の p 要素に描画されるため、フォントサイズはそこに指定する
   （全体ズーム0.8適用後で実効約17px） */
.st-key-qa_float [data-testid="stPopover"] button p {
    font-size: 21px !important;
    font-weight: 700 !important;
    color: white !important;
}
.st-key-qa_float [data-testid="stPopover"] button:hover {
    background: #7E57C2;
    color: white;
}
/* ポップオーバー本体は document.body 直下に描画されるため全体指定
   （このアプリのポップオーバーはQAパネルのみ）。
   ダーク背景＋白文字で、白基調の本文・フォームと一目で区別できるようにする */
[data-testid="stPopoverBody"] {
    /* 本体（.block-container）と同じ0.8ズームを適用して文字サイズを揃える。
       幅はズーム後の見た目で画面の約2/3になるよう 83vw（0.8×83≒66vw）を指定 */
    zoom: 0.8;
    width: 83vw !important;
    max-width: 83vw !important;
    /* 高さ：従来の約2倍。中身が少なくても min-height で確保し、
       Streamlit 既定の maxHeight:70vh の上限も引き上げる（ズーム後の見た目 ≒ 68vh） */
    min-height: 85vh;
    max-height: 110vh !important;
    /* 上端をボタンから少し下げる（ズーム後の見た目 ≒ 58px） */
    margin-top: 72px !important;
    background: #241C3B !important;   /* 紫がかったダーク */
    border: 2px solid #7E57C2 !important;
    border-radius: 12px;
    box-shadow: 0 10px 32px rgba(0,0,0,.45) !important;
}
/* BaseWeb Popover は Body の内側にもう1枚白背景のコンテナ（Inner）を持つため
   透過させ、Body のダーク背景をパネル全面に効かせる */
[data-testid="stPopoverBody"] > div,
[data-testid="stPopoverBody"] [data-testid="stForm"] {
    background: transparent !important;
}
/* フォームの枠線はダーク背景に馴染む淡い白に */
[data-testid="stPopoverBody"] [data-testid="stForm"] {
    border-color: rgba(255,255,255,.30) !important;
}
/* パネル内のテキストは白系に（見出し・説明・キャプションすべて） */
[data-testid="stPopoverBody"] p,
[data-testid="stPopoverBody"] strong,
[data-testid="stPopoverBody"] span,
[data-testid="stPopoverBody"] label {
    color: #F3EFFA !important;
}
[data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] p,
[data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] span {
    color: #C5B8E3 !important;   /* 補足文はやや落とした薄紫 */
}
/* 入力欄は白のまま残し、ダーク背景の中で「書く場所」を浮き上がらせる */
[data-testid="stPopoverBody"] textarea {
    background: #ffffff !important;
    color: #1a1a1a !important;
}
/* パネル内の最新Q&A表示（長い回答はボックス内でスクロール） */
[data-testid="stPopoverBody"] .qa-panel-answer {
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.20);
    border-left: 4px solid #B39DDB;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0 10px 0;
    max-height: 640px;
    overflow-y: auto;
}
/* 文字サイズは 0.8 ズーム適用下で本体の本文（16px）・補足（14px）と揃える */
[data-testid="stPopoverBody"] .qa-panel-q {
    color: #C5B8E3;
    font-size: 14px;
    margin-bottom: 6px;
}
[data-testid="stPopoverBody"] .qa-panel-a {
    color: #F3EFFA;
    font-size: 16px;
    line-height: 1.7;
}
/* 質問履歴クリアボタン：ダーク紫背景で目立つアンバー色 */
[data-testid="stPopoverBody"] .st-key-qa_clear button {
    background: #FFC107;
    border: none;
    font-weight: 700;
}
[data-testid="stPopoverBody"] .st-key-qa_clear button:hover {
    background: #FFD54F;
}
[data-testid="stPopoverBody"] .st-key-qa_clear button p {
    color: #311B92 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}
/* 添付アップローダーもダーク背景に馴染ませる */
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.08) !important;
    border: 1px dashed rgba(255,255,255,.35) !important;
    color: #F3EFFA !important;
}
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] span,
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] small {
    color: #C5B8E3 !important;
}
/* アップローダーの「Upload」ボタン：ダーク紫背景でも読める白ボタン。
   パネル全体の白文字指定に負けないよう、中の文字・アイコンにも紫を明示する */
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] button {
    background: #ffffff !important;
    color: #5E35B1 !important;
    border: 1px solid #B39DDB !important;
    font-weight: 700 !important;
}
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] button p,
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] button span,
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"] {
    color: #5E35B1 !important;
    font-weight: 700 !important;
}
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] button:hover {
    background: #EDE7F6 !important;
    border-color: #9575CD !important;
}

/* 送信ボタンはダーク背景に映える明るめの紫 */
[data-testid="stPopoverBody"] button[kind="primaryFormSubmit"],
[data-testid="stPopoverBody"] button[kind="primary"] {
    background: #7E57C2 !important;
    border-color: #7E57C2 !important;
    color: #ffffff !important;
}
[data-testid="stPopoverBody"] button[kind="primaryFormSubmit"]:hover,
[data-testid="stPopoverBody"] button[kind="primary"]:hover {
    background: #9575CD !important;
    border-color: #9575CD !important;
}

/* QA（質問対応）のUIは右上の質問パネル内で完結する（メイン画面には表示しない） */

.phase-banner {
    background: linear-gradient(135deg, #1565C0, #1976D2);
    color: white; padding: 10px 16px; border-radius: 8px;
    font-weight: 700; font-size: 14px; margin-bottom: 12px;
    /* サイドバーのステップリンクで飛んだ際、固定ヘッダーに隠れないための余白 */
    scroll-margin-top: 110px;
}

.search-log-bubble {
    background: #F5F5F5;
    border-left: 3px solid #90A4AE;
    border-radius: 0 6px 6px 6px;
    padding: 4px 12px;
    margin: 1px 0;
    font-size: 12px;
    color: #546E7A;
    font-family: monospace;
}

.sidebar-card {
    background: white; padding: 10px 14px; border-radius: 8px;
    margin: 4px 0; font-size: 13px;
    box-shadow: 0 1px 3px rgba(0,0,0,.07);
}

/* 再実行中の残存（stale）ボタンを不可視にする。
   重い処理（調査・レポート生成）の間、前サイクルの確認ボタン等が
   薄い色で数分残り続ける・誤クリックされるのを防ぐ。
   display:none で要素全体を消すとページの高さが潰れてスクロール位置が
   先頭に飛ぶため、visibility でレイアウトを保ったまま隠す。 */
div[data-stale="true"] button { visibility: hidden !important; }

/* メイン領域の下パディングを詰める（入力欄の下に広い空白ができるのを防ぐ） */
.block-container { padding-bottom: 2rem !important; }

/* タイトル（h1）まわりの余白を少し詰める：上はコンテナの上パディング、
   下は h1 自体のパディングと直後の区切り線のマージンを縮める */
.block-container { padding-top: 3.5rem !important; }
[data-testid="stHeading"] h1 { padding-top: 0.4rem !important; padding-bottom: 1rem !important; }
.block-container hr { margin-top: 1em; margin-bottom: 1.2em; }

/* 入力欄の英語ヒント（Press Enter to submit form 等）を非表示 */
div[data-testid="InputInstructions"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 定数
# ─────────────────────────────────────────
PHASES = ["1. ヒアリング", "2. 情報整理・分析", "3. 調査方針確認", "4. 調査実施", "5. 調査結果確認", "6. レポート作成", "7. 完了確認"]

# ステップ5は完了マーク（✅）と紛らわしいため「確認」を表す 👀 を使う
PHASE_ICONS = ["💬", "📊", "📋", "🔍", "👀", "📄", "🎉"]

PHASE_INDEX = {
    "hearing": 0, "analysis": 1, "policy_review": 2,
    "searching": 3, "synthesizing": 4, "results_review": 4,
    "reporting": 5, "report_review": 5, "complete": 6,
}

EQUIPMENT_LABELS = {
    "equipment_type":     ("🏭", "設備種別"),
    "installation_place": ("📍", "設置場所"),
    "operation_purpose":  ("🎯", "用途・目的"),
    "scheduled_date":     ("📅", "稼働予定"),
    "chemicals":          ("🧪", "薬品・ガス"),
    "fire_exhaust":       ("🔥", "火気・排気・粉じん"),
    "wastewater":         ("💧", "排水・廃棄物"),
    "noise_vibration":    ("📢", "騒音・振動"),
    "radiation":          ("☢️", "放射線・X線"),
    "construction":       ("🏗️", "建屋改修"),
}

# 内部フィールド名 → 日本語項目名（LLM出力に英字内部名が混ざった場合の表示用置換）
FIELD_NAME_JA = {key: label for key, (_icon, label) in EQUIPMENT_LABELS.items()}
FIELD_NAME_JA["additional_info"] = "その他の情報"

# ヒアリング11項目（表示・確認フォームの順序）
HEARING_FIELDS = list(EQUIPMENT_LABELS.keys()) + ["additional_info"]


def to_ja_field_names(text: str) -> str:
    """LLM出力中の英字内部フィールド名（operation_purpose 等）を日本語項目名に置換する。"""
    for en, ja in FIELD_NAME_JA.items():
        text = text.replace(en, f"「{ja}」")
    return text


def _suggestion_html(html: str) -> str:
    """Google 検索候補（searchEntryPoint）の HTML を iframe 表示用に補正する。
    Google が返すリンクには target 属性がなく、st.iframe の iframe 内で
    遷移しようとして Google 側の X-Frame-Options に拒否され「開かない」ため、
    <base target="_blank"> で全リンクを新しいタブで開かせる。"""
    return '<base target="_blank">' + html



# ─────────────────────────────────────────
# セッション初期化
# ─────────────────────────────────────────
def init():
    defaults = {
        "thread_id":        str(uuid.uuid4()),
        "ui_phase":         "start",   # start | confirm_extract | hearing | interrupt | complete
        "interrupt_data":   None,
        "step_idx":         0,         # 現在のステップ番号（0〜6）
        "review_decisions": {},
        "report_html":      "",
        "display_messages": [],
        "msg_count":        0,         # 表示済みメッセージ数（重複防止）
        "extracted_info":   None,      # 資料から抽出した設備情報（確認前）
        "extract_failed_files": [],    # テキスト抽出できなかったファイル名
        "extract_truncated_files": [], # 文字数上限で一部のみ読み取ったファイル情報
        "expected_questions": 11,      # AIが質問する残り項目数（資料確定分だけ減る）
        "api_key_ok":       bool(
            os.getenv("POC_LLM_API_KEY") or os.getenv("PROD_LLM_API_KEY")
        ),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


# ─────────────────────────────────────────
# 表示用メッセージ追加
# ─────────────────────────────────────────
def add_display(role: str, content: str):
    st.session_state.display_messages.append({
        "role":    role,
        "content": content,
        "time":    datetime.now().strftime("%H:%M"),
    })


def add_step_banner(step_idx: int):
    msgs = st.session_state.display_messages
    # 直近のステップバナーが同じステップなら追加しない
    # （間にAI/systemメッセージが入っても重複バナーを出さない）
    for m in reversed(msgs):
        if m.get("role") == "step_banner":
            if m.get("step") == step_idx:
                return
            break
    msgs.append({"role": "step_banner", "step": step_idx})


def _process_new_msgs(old_count: int, skip_human: bool = True):
    """チェックポイントから新規メッセージのみを display_messages に追加する。
    skip_human=True のとき、先頭の HumanMessage(今回送信したもの)を1件スキップする。"""
    all_msgs = get_all_messages(st.session_state.thread_id)
    start = old_count + (1 if skip_human else 0)
    for m in all_msgs[start:]:
        if isinstance(m, AIMessage) and m.content:
            if getattr(m, "name", None) == "search_progress":
                add_display("search_log", m.content)
            else:
                add_display("ai", m.content)
        elif isinstance(m, ToolMessage) and HEARING_COMPLETE_MARKER in str(m.content):
            # ヒアリング完了の ToolMessage のみ対象。complete_hearing 却下時の
            # 差し戻し ToolMessage では情報整理バナーを出さない
            add_step_banner(1)
            add_display("system", "AIがヒアリング情報を分析し、調査方針を作成しました。調査方針をご確認ください。")
            st.session_state.step_idx = 1   # ステップ2: 情報整理
    st.session_state.msg_count = len(all_msgs)


def _save_phase_idata(idata: dict):
    """フェーズ詳細を session_state に保存し、バナー直下に常時表示できるようにする。
    次の描画（render_messages）で参照されるため、interrupt 設定時に呼ぶ。"""
    phase = idata.get("phase", "")
    if phase == "policy_review":
        st.session_state.policy_idata = idata
    elif phase == "results_review":
        st.session_state.results_idata = idata


# ─────────────────────────────────────────
# LLM 使用量・コストの積算
# ─────────────────────────────────────────
class _LLMUsageCallback(UsageMetadataCallbackHandler):
    """旧 langchain_community.get_openai_callback 互換のトークン・呼び出し回数集計。
    （langchain-community の廃止に伴い langchain_core の使用量集計へ移行。
    単価表は持たないため total_cost は常に 0 で、金額は環境変数の単価か
    Langfuse で計算する）"""

    def __init__(self):
        super().__init__()
        self.successful_requests = 0
        self.total_cost = 0.0

    def on_llm_end(self, response, **kwargs):
        with self._lock:
            self.successful_requests += 1
        super().on_llm_end(response, **kwargs)

    @property
    def prompt_tokens(self) -> int:
        return sum(u.get("input_tokens", 0) for u in self.usage_metadata.values())

    @property
    def completion_tokens(self) -> int:
        return sum(u.get("output_tokens", 0) for u in self.usage_metadata.values())


# コンテキスト変数への登録はモジュール読み込み時に1回だけ行う
# （呼び出しごとに register するとフックが増え続けるため）
_llm_usage_cb_var: ContextVar = ContextVar("llm_usage_callback", default=None)
register_configure_hook(_llm_usage_cb_var, inheritable=True)


@contextmanager
def get_llm_usage_callback():
    """このブロック内の全 LangChain/LangGraph LLM 呼び出しの使用量を集計する。"""
    cb = _LLMUsageCallback()
    token = _llm_usage_cb_var.set(cb)
    try:
        yield cb
    finally:
        _llm_usage_cb_var.reset(token)


def _record_llm_usage(cb, web_cb=None) -> None:
    """get_llm_usage_callback の計測結果を案件単位で積算する。

    web_cb は collect_gemini_usage() の集計結果（このセッションが実行した
    Web検索分のみ）。渡された場合はあわせて積算する。

    Langfuse 設定時はコスト計測を Langfuse に一本化するため、アプリ側の
    概算積算は行わない（サイドバーのコスト表示も出ない）。

    金額は「確かな単価があるとき」＝環境変数 LLM_COST_INPUT_PER_1M /
    LLM_COST_OUTPUT_PER_1M（USD/100万トークン）が明示設定されている場合だけ
    計上する。無い呼び出しは unpriced（単価不明）として件数のみ数え、金額には
    含めない（誤った単価で「それらしい金額」を表示しない）。"""
    if langfuse_enabled():
        return
    _record_web_search_usage(web_cb)
    cost = cb.total_cost
    priced = cost > 0
    if not priced and (cb.prompt_tokens or cb.completion_tokens):
        in_rate  = os.getenv("LLM_COST_INPUT_PER_1M")
        out_rate = os.getenv("LLM_COST_OUTPUT_PER_1M")
        if in_rate and out_rate:
            try:
                cost = (cb.prompt_tokens / 1e6 * float(in_rate)
                        + cb.completion_tokens / 1e6 * float(out_rate))
                priced = True
            except ValueError:
                logger.warning("LLM_COST_*_PER_1M が数値ではないため概算をスキップ")
    usage = st.session_state.setdefault(
        "llm_usage",
        {"prompt": 0, "completion": 0, "cost": 0.0, "calls": 0, "unpriced": 0},
    )
    usage.setdefault("unpriced", 0)
    usage["prompt"]     += cb.prompt_tokens
    usage["completion"] += cb.completion_tokens
    usage["calls"]      += cb.successful_requests
    if priced:
        usage["cost"] += cost
    elif cb.prompt_tokens or cb.completion_tokens:
        usage["unpriced"] += 1


def _record_web_search_usage(web_cb) -> None:
    """このセッションが実行した Gemini Web検索の使用量を積算する。

    web_cb は collect_gemini_usage() の集計結果。プロセス共有カウンタの
    差分ではなく自セッション分だけを受け取るため、複数人が同時に使っても
    他セッションの検索分が混ざらない（従来の差分方式は誤配賦・二重計上した）。

    単価は次の優先順で決める：
    ① 環境変数 GEMINI_COST_INPUT_PER_1M / GEMINI_COST_OUTPUT_PER_1M（明示設定）
    ② 使用モデルが既定の gemini-2.0-flash の場合のみ、内蔵の既定単価（0.10/0.40）
    どちらも該当しない（モデル変更＋単価未設定）場合は金額に計上せず
    unpriced フラグを立てる（誤った単価で計算しない）。
    検索1回あたりの課金（無料枠超過時）は GEMINI_COST_PER_SEARCH で指定する。"""
    # Langfuse 設定時はコスト計測を Langfuse に一本化する（LLM側と同じ方針）
    if not web_cb or langfuse_enabled():
        return
    d_prompt   = web_cb.get("prompt_tokens", 0)
    d_complete = web_cb.get("completion_tokens", 0)
    d_requests = web_cb.get("requests", 0)
    if d_prompt or d_complete or d_requests:
        rates = None
        in_env  = os.getenv("GEMINI_COST_INPUT_PER_1M")
        out_env = os.getenv("GEMINI_COST_OUTPUT_PER_1M")
        model   = os.getenv("GEMINI_WEB_SEARCH_MODEL", "gemini-2.0-flash")
        if in_env and out_env:
            try:
                rates = (float(in_env), float(out_env))
            except ValueError:
                logger.warning("GEMINI_COST_*_PER_1M が数値ではないため概算をスキップ")
        elif model == "gemini-2.0-flash":
            rates = (0.10, 0.40)
        web = st.session_state.setdefault(
            "web_usage",
            {"prompt": 0, "completion": 0, "requests": 0, "cost": 0.0, "unpriced": False},
        )
        web.setdefault("unpriced", False)
        web["prompt"]     += d_prompt
        web["completion"] += d_complete
        web["requests"]   += d_requests
        if rates is not None:
            per_search = float(os.getenv("GEMINI_COST_PER_SEARCH", "0"))
            web["cost"] += (
                d_prompt / 1e6 * rates[0]
                + d_complete / 1e6 * rates[1]
                + d_requests * per_search
            )
        else:
            web["unpriced"] = True


# ─────────────────────────────────────────
# 確認フェーズの QA（質問対応）
# ─────────────────────────────────────────
def run_qa(question: str, fmt: str = "answer", use_web: bool = False) -> None:
    """確認フェーズの質問に回答する。ワークフローの状態（interrupt/checkpointer）は
    一切変更しない。回答は display_messages にのみ積む。
    fmt: "answer"（回答のみ）／ "html"・"docx"・"pdf"・"pptx"（依頼内容から資料ファイルを作成）
    use_web: True の場合、質問文でWeb検索を実行し、結果を回答・資料の根拠に加える"""
    phase = (st.session_state.interrupt_data or {}).get("phase", "")
    label = {
        "policy_review":  "3. 調査方針確認（調査はまだ実施していない）",
        "results_review": "5. 調査結果確認",
        "report_review":  "6. レポート確認",
    }.get(phase) or {
        "start":           "開始前（ヒアリング未開始）",
        "confirm_extract": "資料からの抽出結果の確認中（ヒアリング開始前）",
        "hearing":         "1. ヒアリング中（AIが設備情報を質問している段階。調査は未実施）",
        "complete":        "7. 完了（レポート承認済み）",
    }.get(st.session_state.ui_phase, "確認画面")
    # ヒアリング完了前は state に equipment_info がまだ無いため、
    # 資料から抽出済みの情報があればそれをコンテキストにする
    tid = st.session_state.thread_id
    equipment_info = (
        get_state_value(tid, "equipment_info")
        or st.session_state.get("extracted_info")
        or {}
    )
    # 調査結果：results_review 未到達でも state に結果があれば読み取って渡す
    # （すべて読み取り専用。workflow の state・checkpointer には書き込まない）
    results = st.session_state.get("results_idata")
    if not results:
        law_items = get_state_value(tid, "law_items")
        if law_items:
            results = {
                "summary":          get_state_value(tid, "synthesis_summary") or "",
                "risk_count":       get_state_value(tid, "risk_count") or {},
                "law_items":        law_items,
                "excluded_laws":    get_state_value(tid, "excluded_laws") or [],
                "uncovered_issues": get_state_value(tid, "uncovered_issues") or [],
                "issue_coverage":   get_state_value(tid, "issue_coverage") or {},
                "coverage_check_failed": get_state_value(tid, "coverage_check_failed") or False,
            }
    # 調査で参照した情報源（e-Gov・Web・社内文書のタイトル一覧）
    search_sources = [
        {"title": r.get("title", ""), "source": r.get("source", "")}
        for r in (get_state_value(tid, "search_results") or [])
        if r.get("title")
    ]
    # 質問パネルで添付された資料：テキスト抽出して回答の根拠に使う
    # （抽出結果は (名前, サイズ) でキャッシュし、連続質問での再抽出を防ぐ。
    #  案件本体の設備情報・調査には一切反映しない）
    attached_docs = []
    doc_cache = st.session_state.setdefault("qa_doc_cache", {})
    for f in st.session_state.get("qa_files") or []:
        data = f.getvalue()
        # 同名・同サイズで内容だけ差し替えた場合に古い抽出結果を
        # 再利用しないよう、内容ハッシュをキーにする
        key = (f.name, hashlib.sha256(data).hexdigest())
        if key not in doc_cache:
            try:
                doc_cache[key] = extract_text_from_file(f.name, data)
            except Exception:
                logger.exception("QA添付資料のテキスト抽出に失敗: %s", f.name)
                doc_cache[key] = ""
        if doc_cache[key].strip():
            attached_docs.append((f.name, doc_cache[key]))

    # 質問パネルからの任意Web検索：質問文をそのままクエリにして参考情報を取得する
    # （案件本体の調査とは独立。この検索は後段のLLM集計ブロックの外で実行するため、
    #  ここで使用量を個別に集計してセッションのコストへ積算する）
    qa_web_results = []
    if use_web:
        with collect_gemini_usage() as qa_web_cb:
            try:
                qa_web_results = search_web(question)
            except Exception:
                logger.error("QAのWeb検索でエラー:\n%s", traceback.format_exc())
        _record_web_search_usage(qa_web_cb)

    context = build_qa_context(
        equipment_info,
        st.session_state.get("policy_idata"),
        results,
        label,
        search_sources=search_sources,
        attached_docs=attached_docs,
        web_results=qa_web_results,
    )
    # 直近の文脈：メイン画面の会話＋パネル内のQAやり取り
    # （QAは display_messages に積まないため、qa_history から補う。
    #  表示用の履歴は全件残し、LLMに渡すのは直近6往復のみ。エラー回答は渡さない）
    qa_hist = st.session_state.get("qa_history", [])
    hist_msgs = list(st.session_state.display_messages)
    for x in qa_hist[-6:]:
        if x["a"].startswith("⚠️"):
            continue
        hist_msgs.append({"role": "user", "content": x["q"]})
        hist_msgs.append({"role": "qa", "content": x["a"]})
    history = build_qa_history(hist_msgs)

    try:
        with get_llm_usage_callback() as cb, collect_gemini_usage() as web_cb, trace_run(
            "qa" if fmt == "answer" else "qa_doc",
            session_id=st.session_state.thread_id, input=question,
        ) as lf:
            try:
                if fmt == "answer":
                    answer = answer_question(question, context, history, config=lf)
                    entry = {"q": question, "a": answer}
                else:
                    title, md = generate_document(
                        question, context, history, config=lf,
                        slides=(fmt == "pptx"),
                    )
                    file_name, data, mime = build_doc_file(fmt, title, md)
                    entry = {
                        "q": question,
                        "a": f"📎 資料を作成しました：**{title}**\n"
                             f"下のボタンからダウンロードできます（この履歴を消すまで残ります）。",
                        "file": {"name": file_name, "data": data, "mime": mime},
                    }
            finally:
                _record_llm_usage(cb, web_cb)
        # 回答は質問パネル内にのみ表示する（メイン画面の会話には積まない）
        st.session_state.qa_history = qa_hist + [entry]
    except Exception:
        logger.error("QA でエラー:\n%s", traceback.format_exc())
        st.session_state.qa_history = qa_hist + [{
            "q": question,
            "a": "⚠️ 質問への回答中にエラーが発生しました。お手数ですが、もう一度お試しください。"
                 if fmt == "answer" else
                 "⚠️ 資料の作成中にエラーが発生しました。お手数ですが、もう一度お試しください"
                 "（PDFで失敗する場合はHTML・Wordもお試しください）。",
        }]


def render_qa_input():
    """全フェーズ共通の質問パネル。画面右上（タイトル行の高さ）に固定表示
    （CSS .st-key-qa_float）し、決定ボタン（案件を進める操作）と視覚的に分離する。"""
    with st.container(key="qa_float"):
        with st.popover("💬 AIに質問"):
            qa_hist = st.session_state.get("qa_history", [])
            # 見出し行の右上に履歴クリアボタンを置く（履歴がある時だけ表示）
            hc1, hc2 = st.columns([3, 1.3], vertical_alignment="center")
            with hc1:
                st.markdown("**❓ わからないことを、いつでもAIに質問できます**")
            with hc2:
                # ボタン幅は列の半分に抑え、右端に寄せる
                _sp, hb = st.columns([1, 1])
                with hb:
                    if qa_hist and st.button(
                        "🗑️ 質問履歴クリア", key="qa_clear", use_container_width=True,
                        help="今までの質問・回答をすべて消します",
                    ):
                        st.session_state.qa_history = []
                        st.rerun()
            st.caption(
                "質問しても案件は進みません（ヒアリングへの回答・承認・再調査は"
                "画面内の入力欄・ボタンから）。回答はこのパネル内に表示されます（参考情報）。"
            )
            # これまでの質問と回答を古い順にすべて表示する（クリアするまで残る）
            for i, x in enumerate(qa_hist):
                q_html = html_lib.escape(x["q"]).replace("\n", "<br>")
                a_html = html_lib.escape(x["a"]).replace("\n", "<br>")
                a_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', a_html)
                st.markdown(
                    f'<div class="qa-panel-answer">'
                    f'<div class="qa-panel-q">Q. {q_html}</div>'
                    f'<div class="qa-panel-a">{a_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # 資料作成の依頼には、生成ファイルのダウンロードボタンを付ける
                f = x.get("file")
                if f:
                    st.download_button(
                        label=f"⬇️ {f['name']}",
                        data=f["data"],
                        file_name=f["name"],
                        mime=f["mime"],
                        key=f"qa_dl_{i}",
                        type="primary",
                    )

            # 送信された質問はこのパネル内で処理する（処理中スピナーは履歴の下＝
            # 最新の位置に表示。メイン画面には何も出さない）
            pending = st.session_state.pop("pending_qa_question", None)
            if pending:
                p_q   = pending["q"]
                p_fmt = pending.get("fmt", "answer")
                p_web = bool(pending.get("web"))
                q_html = html_lib.escape(p_q).replace("\n", "<br>")
                st.markdown(
                    f'<div class="qa-panel-answer">'
                    f'<div class="qa-panel-q">Q. {q_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                spinner_msg = (
                    "💬 質問に回答しています..." if p_fmt == "answer"
                    else "📄 資料を作成しています...（1分ほどかかる場合があります）"
                )
                if p_web:
                    spinner_msg = "🌐 Web検索中... → " + spinner_msg
                with st.spinner(spinner_msg):
                    run_qa(p_q, p_fmt, use_web=p_web)
                st.rerun()   # 処理中表示を消し、上の履歴表示に切り替える
            # 添付はフォームの外に置く（clear_on_submit で消えず、連続質問で使い回せる）
            st.file_uploader(
                "📎 資料を添付して質問（任意。回答にのみ使用し、案件・調査には反映されません）",
                type=["pdf", "docx", "xlsx", "xlsm", "pptx", "txt"],
                accept_multiple_files=True,
                key="qa_files",
            )
            with st.form("qa_form", clear_on_submit=True):
                q = st.text_area(
                    "質問内容",
                    placeholder="例：「特定施設」とはなんですか？　添付資料のこの装置に届出は必要ですか？\n"
                                "（資料作成の例：この案件の対応事項を上長向けの説明資料にまとめて）",
                    height=80,
                    label_visibility="collapsed",
                )
                fmt_labels = {
                    "💬 回答のみ":   "answer",
                    "🌐 HTML資料":  "html",
                    "📝 Word資料":  "docx",
                    "📕 PDF資料":   "pdf",
                    "📊 PowerPoint資料": "pptx",
                }
                fmt_choice = st.radio(
                    "出力形式",
                    list(fmt_labels.keys()),
                    horizontal=True,
                    help="「HTML・Word・PDF・PowerPoint」を選ぶと、入力内容を依頼として"
                         "資料ファイルを作成し、ダウンロードできます（案件・調査には反映されません）。",
                )
                use_web = st.checkbox(
                    "🌐 Web検索を使う（最新の公開情報を検索して回答・資料の根拠に加える）",
                    value=False,
                    help="質問文でWeb検索（Google検索）を実行し、その結果も参考にします。"
                         "案件本体の調査には反映されません。少し時間と費用がかかります。",
                )
                if st.form_submit_button("質問する ➤", type="primary", use_container_width=True):
                    if q.strip():
                        st.session_state.pending_qa_question = {
                            "q": q.strip(), "fmt": fmt_labels[fmt_choice], "web": use_web,
                        }
                        st.rerun()


# ─────────────────────────────────────────
# LangGraph 呼び出し（ヒアリング中）
# ─────────────────────────────────────────
def _rollback_predicted_analysis():
    """11問カウントの予測で先行表示した「2. 情報整理」バナー・ステップを取り消す。
    （AIが聞き直しや挨拶を挟んでカウントがずれた場合の自己修復）"""
    msgs = st.session_state.display_messages
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "step_banner":
            if msgs[i].get("step") == 1:
                msgs.pop(i)
            break
    if st.session_state.step_idx == 1:
        st.session_state.step_idx = 0


def invoke_hearing(user_text: str):
    old_count = st.session_state.msg_count
    config = get_config()

    # AIの質問が残り項目数（資料からの確定分だけ減る。全問手入力なら11）に
    # 達していれば、次のinvokeでヒアリング完了→情報整理に移行する
    ai_count = sum(1 for m in st.session_state.display_messages if m.get("role") == "ai")
    hearing_ending = ai_count >= st.session_state.get("expected_questions", 11)
    # 処理中の表示はスピナーのみ（バナーを直接描画すると入力フォームの下に
    # 割り込んで表示が崩れるため）。完了後の rerun で会話フロー内の正しい
    # 位置にステップバナーが表示される。

    try:
        with st.spinner("情報整理・分析中..." if hearing_ending else "AIが考えています..."):
            with get_llm_usage_callback() as cb, collect_gemini_usage() as web_cb, trace_run(
                "hearing", session_id=st.session_state.thread_id, input=user_text,
            ) as lf:
                try:
                    workflow.invoke(
                        {"messages": [HumanMessage(content=user_text)]},
                        config={**config, **lf},
                    )
                finally:
                    # エラー時も、そこまでに消費したトークンを計上する
                    _record_llm_usage(cb, web_cb)
    except Exception:
        logger.error("invoke_hearing でエラー:\n%s", traceback.format_exc())
        add_display("system", "⚠️ AIとの通信でエラーが発生しました。お手数ですが、同じ内容をもう一度送信してください。")
        if not get_state_value(st.session_state.thread_id, "hearing_complete"):
            _rollback_predicted_analysis()
        return

    _process_new_msgs(old_count, skip_human=True)

    idata = get_interrupt_data(st.session_state.thread_id)
    if idata:
        st.session_state.interrupt_data = idata
        st.session_state.ui_phase = "interrupt"
        st.session_state.step_idx = PHASE_INDEX.get(idata.get("phase", ""), 1)
        add_step_banner(st.session_state.step_idx)
        _save_phase_idata(idata)
    elif not get_state_value(st.session_state.thread_id, "hearing_complete"):
        # ヒアリング継続中：カウント予測が外れて先行表示した
        # 「2. 情報整理」バナーがあれば取り消す（workflow の状態を一次情報源にする）
        _rollback_predicted_analysis()


# ─────────────────────────────────────────
# 資料アップロード → 抽出 → 確認
# ─────────────────────────────────────────
def start_hearing_plain():
    """資料なしの通常ヒアリングを開始する。"""
    add_step_banner(0)
    add_display("user", "法令確認・届出施設確認を開始します。")
    st.session_state.step_idx = 0
    invoke_hearing("法令確認・届出施設確認を開始します。設備情報のヒアリングをお願いします。")
    st.session_state.ui_phase = "hearing"
    st.rerun()


def start_with_documents(files) -> bool:
    """アップロード資料からテキストを抽出し、LLMで11項目を構造化抽出して
    確認画面（confirm_extract）へ進む。1件も抽出できなければ False を返す
    （呼び出し側で通常ヒアリングにフォールバックする）。"""
    doc_texts, failed, truncated = [], [], []
    extracted = None
    with st.spinner("📄 資料から設備情報を抽出中...（資料が多いと1分程度かかります）"):
        for f in files:
            text = extract_text_from_file(f.name, f.getvalue())
            if text.strip():
                doc_texts.append((f.name, text))
            else:
                failed.append(f.name)
        if doc_texts:
            try:
                with get_llm_usage_callback() as cb, collect_gemini_usage() as web_cb, trace_run(
                    "doc_extraction",
                    session_id=st.session_state.thread_id,
                    input=[name for name, _t in doc_texts],
                ) as lf:
                    try:
                        extracted, truncated = extract_equipment_info(doc_texts, config=lf)
                    finally:
                        _record_llm_usage(cb, web_cb)
            except Exception:
                logger.error("資料からの情報抽出でエラー:\n%s", traceback.format_exc())

    st.session_state.extract_failed_files = failed
    st.session_state.extract_truncated_files = truncated
    if extracted:
        st.session_state.extracted_info = extracted
        st.session_state.ui_phase = "confirm_extract"
        return True
    return False


def render_extract_confirm():
    """資料から抽出した11項目をユーザーに確認・修正してもらう画面。"""
    info = st.session_state.extracted_info or {}
    failed = st.session_state.get("extract_failed_files") or []
    if failed:
        st.warning(
            "⚠️ 次のファイルはテキストを抽出できませんでした（スキャン画像のPDF等）："
            + "、".join(failed)
        )
    truncated = st.session_state.get("extract_truncated_files") or []
    if truncated:
        st.warning(
            "⚠️ 文字数上限のため、次のファイルは一部のみ読み取りました。"
            "後半に記載の情報（数量・容量等）が抽出されていない可能性があるため、"
            "下記の内容をご確認ください："
            + "、".join(
                f'{t["name"]}（{t["used"]:,}/{t["total"]:,}字）' for t in truncated
            )
        )

    filled_count = sum(
        1 for k in HEARING_FIELDS if (info.get(k) or {}).get("value", "").strip()
    )
    st.markdown(
        f'<div class="system-bubble">📄 資料から設備情報を抽出しました'
        f'（{filled_count} / {len(HEARING_FIELDS)}項目）。'
        f'内容が正しいか確認し、誤りがあれば修正してください。<br>'
        f'<b>空欄のままの項目は、この後AIが1項目ずつ質問します。</b>'
        f'（各項目の ❓ アイコンで資料の根拠を確認できます）</div>',
        unsafe_allow_html=True,
    )

    with st.form("extract_confirm_form"):
        values = {}
        for key in HEARING_FIELDS:
            icon, label = EQUIPMENT_LABELS.get(key, ("📝", FIELD_NAME_JA[key]))
            field = info.get(key) or {}
            evidence = field.get("evidence", "")
            help_text = (
                f"資料の根拠：{evidence}" if evidence
                else "資料に記載が見つかりませんでした（空欄のままだとAIが質問します）"
            )
            # 抽出済み／未記入が一目で分かるようラベルにバッジを付ける
            status = (
                "✅ 抽出済み" if field.get("value", "").strip()
                else "❓ 未記入（この後AIが質問）"
            )
            values[key] = st.text_input(
                f"{icon} {label}　{status}",
                value=field.get("value", ""),
                help=help_text,
                key=f"extract_{key}",
            )
        submitted = st.form_submit_button(
            "✅　この内容で確定してヒアリングを開始する",
            type="primary", use_container_width=True,
        )
    if submitted:
        confirm_extracted_and_start(values)


def confirm_extracted_and_start(values: dict):
    """確認・修正済みの抽出情報を確定情報としてAIに渡し、
    未記入の項目だけを質問するヒアリングを開始する。"""
    filled = {k: v.strip() for k, v in values.items() if v.strip()}
    missing = [k for k in HEARING_FIELDS if k not in filled]
    # AIが質問するのは未記入項目のみ（完了予測・スピナー表示に使う）
    st.session_state.expected_questions = len(missing)

    lines = [f"・{FIELD_NAME_JA[k]}：{filled.get(k, '（未記入）')}" for k in HEARING_FIELDS]
    listing = "\n".join(lines)

    add_step_banner(0)
    add_display("user", "資料から抽出した設備情報を確認・確定しました。\n" + listing)
    st.session_state.step_idx = 0
    st.session_state.ui_phase = "hearing"

    if not missing:
        # 全項目確定済み：質問なしで complete_hearing → 情報整理まで一気に進む
        st.session_state.step_idx = 1
        add_step_banner(1)

    prompt = (
        "法令確認・届出施設確認を開始します。\n"
        "以下は関連資料から抽出し、担当者が確認・修正した設備情報です。\n"
        "値が記載されている項目は確定情報として扱い、再質問しないでください。\n"
        f"「（未記入）」の項目が{len(missing)}件あります。そのすべてについて、"
        "1項目ずつ質問して回答を得てください。\n"
        "全項目の回答が揃うまで complete_hearing を呼ばないでください。\n"
        "すべて記入済みの場合のみ、質問せず直ちに complete_hearing を呼び出してください。\n\n"
        + listing
    )
    invoke_hearing(prompt)
    st.rerun()


def submit_hearing_answer(user_text: str):
    """ヒアリングの回答送信を一元処理する。
    最終問の回答（次が情報整理）のときは、重い処理に入る前に一旦
    サイドバー・バナーをステップ2に更新してから rerun し、次サイクルで
    情報整理を実行する。これにより「情報整理中」の間もサイドバー・バナーが
    ステップ2を正しく表示する。"""
    add_display("user", user_text)
    ai_count = sum(1 for m in st.session_state.display_messages if m.get("role") == "ai")
    if ai_count >= st.session_state.get("expected_questions", 11):
        # 情報整理フェーズへ移行：先に画面表示を更新してから処理する
        st.session_state.step_idx = 1
        add_step_banner(1)
        st.session_state.pending_analysis_text = user_text
    else:
        invoke_hearing(user_text)
    st.rerun()


# ─────────────────────────────────────────
# LangGraph 呼び出し（interrupt 再開）
# ─────────────────────────────────────────
def _rollback_resume_banner(current_phase: str):
    """再開処理の失敗時、先行表示した「実行中」バナー（調査実施/レポート作成）を
    取り消し、ステップ表示を元の確認フェーズに戻す。"""
    msgs = st.session_state.display_messages
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "step_banner":
            if msgs[i].get("step") in (3, 5):
                msgs.pop(i)
            break
    st.session_state.step_idx = PHASE_INDEX.get(current_phase, st.session_state.step_idx)


def resume_graph(decision):
    old_count = st.session_state.msg_count
    config = get_config()

    # 再調査（reinvestigate）依頼や方針承認は検索フェーズに入るのでライブ表示する
    current_phase = (st.session_state.interrupt_data or {}).get("phase", "")
    will_search = (
        current_phase == "policy_review"
        or (isinstance(decision, str) and decision.startswith("reinvestigate"))
    )

    # 現フェーズに応じて「実行中」バナーを先行表示
    if will_search:
        add_step_banner(3)  # ステップ4: 調査実施
    elif current_phase == "results_review":
        add_step_banner(5)  # ステップ6: レポート作成

    if will_search:
        # 調査フェーズは各ステップの進捗をライブ表示（stream_mode="custom"）
        status = st.status("🔎 AI調査を開始しています...", expanded=True)
        try:
            with get_llm_usage_callback() as cb, collect_gemini_usage() as web_cb, trace_run(
                f"search:{current_phase or 'resume'}",
                session_id=st.session_state.thread_id,
                input=decision,
            ) as lf:
                try:
                    for chunk in workflow.stream(
                        Command(resume=decision), config={**config, **lf},
                        stream_mode="custom",
                    ):
                        entry = chunk.get("progress") if isinstance(chunk, dict) else None
                        if entry:
                            status.update(label=f"AI調査中... {entry}")
                            status.write(entry)
                finally:
                    # エラー時も、そこまでに消費したトークンを計上する
                    _record_llm_usage(cb, web_cb)
        except Exception:
            logger.error("調査フェーズ (workflow.stream) でエラー:\n%s", traceback.format_exc())
            status.update(label="⚠️ 調査中にエラーが発生しました", state="error", expanded=True)
            add_display("system", "⚠️ 調査中にエラーが発生しました。もう一度お試しください。")
            _rollback_resume_banner(current_phase)
            return
        else:
            status.update(label="✅ AI調査が完了しました", state="complete", expanded=False)
    else:
        try:
            with st.spinner("処理中..."):
                with get_llm_usage_callback() as cb, collect_gemini_usage() as web_cb, trace_run(
                    f"resume:{current_phase or 'resume'}",
                    session_id=st.session_state.thread_id,
                    input=decision,
                ) as lf:
                    try:
                        workflow.invoke(Command(resume=decision), config={**config, **lf})
                    finally:
                        _record_llm_usage(cb, web_cb)
        except Exception:
            logger.error("処理フェーズ (workflow.invoke) でエラー:\n%s", traceback.format_exc())
            add_display("system", "⚠️ 処理中にエラーが発生しました。もう一度お試しください。")
            _rollback_resume_banner(current_phase)
            return

    _process_new_msgs(old_count, skip_human=False)

    idata = get_interrupt_data(st.session_state.thread_id)
    if idata:
        st.session_state.interrupt_data = idata
        st.session_state.ui_phase = "interrupt"
        st.session_state.step_idx = PHASE_INDEX.get(idata.get("phase", ""), 0)
        add_step_banner(st.session_state.step_idx)
        _save_phase_idata(idata)
    else:
        rhtml = get_state_value(st.session_state.thread_id, "report_html")
        if rhtml:
            st.session_state.report_html = rhtml
            # レポートを outputs/ に自動保存（「新しい案件」で消える事故を防ぐ）
            try:
                case_id = get_state_value(st.session_state.thread_id, "case_id") or "case"
                os.makedirs("outputs", exist_ok=True)
                out_path = os.path.join(
                    "outputs",
                    f"{datetime.now().strftime('%Y%m%d_%H%M')}_{case_id}_法令確認レポート.html",
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(rhtml)
                st.session_state.report_saved_path = out_path
            except Exception:
                logger.error("レポートの自動保存に失敗:\n%s", traceback.format_exc())
                st.session_state.report_saved_path = ""
        st.session_state.ui_phase = "complete"
        st.session_state.interrupt_data = None
        st.session_state.step_idx = 6
        add_step_banner(6)


def request_resume(decision):
    """interrupt 再開をリクエストする。調査・レポート生成などの重い処理に
    入る前に、先にステップ表示（サイドバー・バナー）を進めてから rerun し、
    次サイクルで実際の処理を実行する。これにより処理中もサイドバーが正しい
    ステップを示し、確認ボタン（ピンク）が処理中に残らない。"""
    current_phase = (st.session_state.interrupt_data or {}).get("phase", "")
    will_search = (
        current_phase == "policy_review"
        or (isinstance(decision, str) and decision.startswith("reinvestigate"))
    )
    if will_search:
        st.session_state.step_idx = 3   # 4. 調査実施
        add_step_banner(3)
    elif current_phase == "results_review":
        st.session_state.step_idx = 5   # 6. レポート作成
        add_step_banner(5)
    st.session_state.pending_resume_decision = decision
    st.rerun()


# ─────────────────────────────────────────
# 共通ステップバナー
# ─────────────────────────────────────────
def render_step_banner():
    if st.session_state.ui_phase == "start":
        return
    idx  = st.session_state.step_idx
    icon = PHASE_ICONS[idx]
    name = PHASES[idx]
    st.markdown(
        f'<div class="phase-banner">{icon}　{name}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
# メッセージ描画
# ─────────────────────────────────────────
def render_messages():
    msgs = st.session_state.display_messages
    # 各ステップの「最後の」バナー位置を求める。詳細はそこにだけアンカー表示する
    # （再調査ループで同じステップのバナーが複数あっても、最新の位置に表示される）
    last_banner_pos: dict = {}
    for i, m in enumerate(msgs):
        if m.get("role") == "step_banner":
            last_banner_pos[m["step"]] = i

    for pos, msg in enumerate(msgs):
        role = msg["role"]

        if role == "step_banner":
            idx  = msg["step"]
            icon = PHASE_ICONS[idx]
            name = PHASES[idx]
            # サイドバーのステップリンクの飛び先。同じステップのバナーが複数ある
            # 場合（再調査ループ）は最後のバナーだけに付け、id の重複を避ける
            anchor = f' id="step-anchor-{idx}"' if last_banner_pos.get(idx) == pos else ""
            st.markdown(
                f'<div class="phase-banner"{anchor}>{icon}　{name}</div>',
                unsafe_allow_html=True,
            )
            # 各フェーズの詳細を、そのステップの最後のバナー直下に常時アンカー表示する
            # （後続フェーズに進んでも消えず、再調査後は最新バナーの下に移動する）
            if last_banner_pos.get(idx) != pos:
                continue
            phase = (st.session_state.interrupt_data or {}).get("phase", "")
            if idx == 2 and st.session_state.get("policy_idata"):
                # 3. 調査方針確認 の直下：調査方針の内容（方針確認中のみ展開）
                render_policy_detail(
                    st.session_state.policy_idata,
                    expanded=(phase == "policy_review"),
                )
            elif idx == 4 and st.session_state.get("results_idata"):
                # 5. 調査結果確認 の直下：調査結果の詳細
                render_results_detail(st.session_state.results_idata)
            continue

        if role == "search_log":
            st.markdown(
                f'<div class="search-log-bubble">{html_lib.escape(msg["content"])}</div>',
                unsafe_allow_html=True,
            )
            continue

        # HTMLエスケープしてから改行・太字のみ変換（入力に <, > 等が含まれても崩れない）
        content = html_lib.escape(msg["content"]).replace("\n", "<br>")
        content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
        t       = msg["time"]
        if role == "ai":
            st.markdown(
                f'<div class="ai-bubble">'
                f'<div class="ai-header">⚖️ 法令確認AI　{t}</div>'
                f'{content}</div>',
                unsafe_allow_html=True,
            )
        elif role == "user":
            st.markdown(
                f'<div class="user-bubble">'
                f'<div class="user-header">👤 あなた　{t}</div>'
                f'<div class="user-body">{content}</div></div>',
                unsafe_allow_html=True,
            )
        elif role == "system":
            st.markdown(
                f'<div class="system-bubble">🔄 {content}</div>',
                unsafe_allow_html=True,
            )
        st.write("")



# ─────────────────────────────────────────
# 入力エリア描画
# ─────────────────────────────────────────
def render_input():
    ui    = st.session_state.ui_phase
    idata = st.session_state.interrupt_data

    # ── 重い再開処理（調査・レポート生成）：表示更新済みのここで実行 ──
    # 確認ボタンを描画する前に処理するため、処理中はボタンが消える。
    # 各フェーズの詳細はバナー直下（上部）に常時表示されるため、ここでの再掲は不要。
    if "pending_resume_decision" in st.session_state:
        decision = st.session_state.pop("pending_resume_decision")
        resume_graph(decision)
        st.rerun()

    # ── 開始前 ──
    if ui == "start":
        if not st.session_state.api_key_ok:
            mode = os.getenv("LLM_MODE", "poc")
            key_name = "POC_LLM_API_KEY" if mode != "prod" else "PROD_LLM_API_KEY"
            st.error(f"⚠️ {key_name} が設定されていません。`.env` ファイルを作成してAPIキーを設定してください。")
            st.code("copy .env.example .env\n# .env を編集して LLM_MODE と API キーを設定")
            return
        st.info(
            "👇 仕様書・設備リストなどの関連資料があればアップロードしてください。"
            "AIが資料から設備情報を抽出し、ヒアリングでの手入力を減らせます。"
            "資料がなくてもそのまま開始できます。"
        )
        st.caption(
            "💡 サイドバーの「📁 社内文書」に社内規定・過去の届出事例を登録しておくと、"
            "AIが調査時に社内文書も検索します。また、レポートを承認した案件は"
            "「📚 過去の調査事例」に蓄積され、次回の類似案件で自動的に参照されます（使うほど賢くなります）。"
        )
        files = st.file_uploader(
            "関連資料（PDF / Word / Excel / PowerPoint / テキスト・複数可）",
            type=["pdf", "docx", "xlsx", "xlsm", "pptx", "txt"],
            accept_multiple_files=True,
        )
        if st.button("▶️　法令確認・届出施設確認を開始する", type="primary", use_container_width=True):
            if files:
                if start_with_documents(files):
                    st.rerun()
                # 全ファイルの抽出に失敗：通常ヒアリングにフォールバック
                add_display(
                    "system",
                    "⚠️ 資料から設備情報を抽出できなかったため（スキャン画像のPDF等）、"
                    "通常のヒアリングを開始します。",
                )
            start_hearing_plain()

    # ── 資料抽出結果の確認 ──
    elif ui == "confirm_extract":
        render_extract_confirm()

    # ── ヒアリング中 ──
    elif ui == "hearing":
        # 情報整理の重い処理は、サイドバー・バナーをステップ2に更新した後の
        # このサイクルで実行する（入力フォームは描画せずスピナーのみ表示）
        if st.session_state.get("pending_analysis_text"):
            txt = st.session_state.pop("pending_analysis_text")
            invoke_hearing(txt)
            st.rerun()

        # 残り質問数の表示（先が見えるようにする）
        expected = st.session_state.get("expected_questions", 11)
        asked = sum(1 for m in st.session_state.display_messages if m.get("role") == "ai")
        if expected > 0 and asked > 0:
            current = min(asked, expected)
            st.progress(
                current / expected,
                text=f"📝 質問 {current} / {expected} 項目（残り {expected - current} 項目）",
            )

        with st.form("hearing_form", clear_on_submit=True):
            txt = st.text_area(
                "AIへの回答を入力してください",
                placeholder="こちらに記入してください。",
                height=80,
            )
            submitted = st.form_submit_button("送信 ➤", type="primary", use_container_width=True)

        st.caption("よく使う選択肢：")
        cols = st.columns(5)
        clicked = None
        for i, opt in enumerate(["あり", "なし", "不明", "未定", "確認中"]):
            if cols[i].button(opt, key=f"q_{opt}", use_container_width=True):
                clicked = opt

        # 送信処理は全ウィジェットの描画後に行う。クリックハンドラ内（描画途中）で
        # 重いLLM処理を実行すると、処理中に未描画の後続ボタンが消えてしまう
        if submitted and txt.strip():
            submit_hearing_answer(txt.strip())
        elif clicked:
            submit_hearing_answer(clicked)

    # ── interrupt: 調査内容確認 ──
    elif ui == "interrupt" and idata and idata.get("phase") == "policy_review":
        render_policy_review(idata)

    # ── interrupt: 結果レビュー ──
    elif ui == "interrupt" and idata and idata.get("phase") == "results_review":
        st.session_state.results_idata = idata  # ステップ6でも再表示できるよう保存
        render_results_review(idata)

    # ── interrupt: レポートレビュー ──
    elif ui == "interrupt" and idata and idata.get("phase") == "report_review":
        render_report_review(idata)

    # ── 完了 ──
    elif ui == "complete":
        st.success("✅ 全プロセスが完了しました！")
        c1, c2 = st.columns(2)
        if st.session_state.get("confirm_new_case"):
            with c1:
                st.warning("この画面の内容はクリアされます。よろしいですか？")
                if st.button("はい、新しい案件を開始する", type="primary", use_container_width=True):
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.rerun()
                if st.button("キャンセル", use_container_width=True):
                    st.session_state.confirm_new_case = False
                    st.rerun()
        elif c1.button("🔄 新しい案件を開始する", use_container_width=True):
            st.session_state.confirm_new_case = True
            st.rerun()
        if st.session_state.report_html:
            from datetime import datetime
            file_name = datetime.now().strftime("%Y%m%d_%H%M") + "_法令･届出施設確認サポートAI作成レポート.html"
            c2.download_button(
                label="⬇️ レポートをダウンロード",
                data=st.session_state.report_html.encode("utf-8"),
                file_name=file_name,
                mime="text/html",
                use_container_width=True,
            )

    # ── 全フェーズ共通：右下固定の質問パネル（いつでも質問できる） ──
    render_qa_input()


# ─────────────────────────────────────────
# 調査内容確認 UI
# ─────────────────────────────────────────
def render_policy_detail(idata: dict, expanded: bool = True):
    """調査方針確認の内容（分析概要・調査項目・不明情報・調査方針）を描画。
    ボタンは含まない。expanded=False で再掲（折りたたみ）表示にする。"""
    with st.expander("📋 ヒアリング情報の分析結果概要", expanded=expanded):
        st.write(to_ja_field_names(idata.get("analysis_summary", "")))

    with st.expander("🔍 調査が必要な項目", expanded=expanded):
        for issue in idata.get("issues", []):
            st.write(f"● {to_ja_field_names(issue)}")

    if idata.get("unknown_items"):
        items_html = "".join(
            f'<li style="margin-bottom:7px;line-height:1.65;">{html_lib.escape(to_ja_field_names(u))}</li>'
            for u in idata["unknown_items"]
        )
        st.markdown(
            f'<div style="border:1px solid #FFB74D;border-left:6px solid #FB8C00;'
            f'border-radius:8px;background:#FFF8E1;padding:14px 18px;margin:10px 0;">'
            f'<div style="color:#E65100;font-weight:700;font-size:16px;margin-bottom:6px;">'
            f'⚠️ 調査前に確認した方がよい不明・未定情報</div>'
            f'<div style="color:#C62828;font-weight:700;font-size:16px;margin-bottom:10px;">'
            f'下記が未確定だと必要な法令確認・届出確認が漏れる恐れがあります。'
            f'「調査前に追記する」から補足してから調査を開始することを推奨します。</div>'
            f'<ul style="margin:0;padding-left:20px;color:#4E342E;font-size:16px;">{items_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander("📌 調査方針", expanded=expanded):
        st.write(to_ja_field_names(idata.get("search_plan", "")))
        st.caption("検索キーワード（e-Gov法令API）: " + " / ".join(idata.get("search_keywords", [])))
        st.caption("🌐 AIWeb検索（自動実施） ● 横浜市・神奈川県の条例・規制（公式サイト） ● 省庁ガイドライン・FAQ（厚生労働省・消防庁・環境省・国土交通省・経済産業省）")


def _build_attachment_note(files, total_cap: int = 8000) -> tuple:
    """添付資料からテキストを抽出し、AIへの追記指示に含める抜粋文字列を作る。
    追記はその後の全プロンプトに毎回入るため、合計 total_cap 字までに抜粋する。
    戻り値: (抜粋テキスト, 読み取れなかったファイル名リスト)"""
    doc_note = ""
    failed = []
    if files:
        # 残り予算を後続ファイル数で均等配分し、合計を必ず total_cap 以下に抑える
        # （固定の per_cap 方式ではファイル数に比例して上限を超過していた）
        remaining = total_cap
        n = len(files)
        for idx, f in enumerate(files):
            try:
                txt = extract_text_from_file(f.name, f.getvalue())
            except Exception:
                logger.exception("添付資料のテキスト抽出に失敗: %s", f.name)
                txt = ""
            if txt.strip():
                per_cap = max(1, remaining // (n - idx))
                excerpt = txt.strip()[:per_cap]
                remaining -= len(excerpt)
                doc_note += f"\n\n【添付資料「{f.name}」の内容抜粋】\n" + excerpt
            else:
                failed.append(f.name)
    return doc_note, failed


def _notify_unreadable_files(failed_files) -> None:
    if failed_files:
        add_display(
            "system",
            "⚠️ テキストを読み取れなかった資料（スキャン画像のPDF等）："
            + "、".join(failed_files),
        )


def render_policy_review(idata: dict):
    st.session_state.policy_idata = idata  # 後続フェーズでも再表示できるよう保存
    # 調査方針の内容は「3. 調査方針確認」バナー直下（上部）に常時表示される。
    # ここでは操作ボタンのみを置く。
    st.info(
        "AIがヒアリング情報を分析し、調査方針を作成しました。"
        "上に表示された調査方針の内容をご確認のうえ、問題なければ"
        "「この方針で調査を開始する」を押してください。"
    )
    st.caption(
        "💡 内容が難しい場合は、そのまま承認して進めて大丈夫です。"
        "調査後の「結果確認」「レポート確認」でも内容の確認・修正・再調査を依頼できます。"
    )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if "show_note_form" not in st.session_state:
            st.session_state.show_note_form = False
        if st.button("✏️　調査前に追記する", type="primary", use_container_width=True, key="toggle_note"):
            st.session_state.show_note_form = not st.session_state.show_note_form
            st.rerun()
        if st.session_state.get("show_note_form"):
            with st.form("policy_note"):
                note = st.text_area("追記内容：", placeholder="例：消防署は○○消防署に絞ってください")
                note_files = st.file_uploader(
                    "📎 追加資料（任意）。内容を読み取り、追記指示と一緒に調査の参考情報としてAIに渡します",
                    type=["pdf", "docx", "xlsx", "xlsm", "pptx", "txt"],
                    accept_multiple_files=True,
                    key="policy_note_files",
                )
                if st.form_submit_button("✅　追記して調査を開始する", type="primary", use_container_width=True):
                    st.session_state.show_note_form = False
                    with st.spinner("添付資料を読み取り中..."):
                        doc_note, failed_files = _build_attachment_note(note_files)
                    combined = (note.strip() + doc_note).strip()
                    disp = f"調査方針を承認しました。追記：{note.strip()}"
                    if note_files:
                        disp += f"（添付資料：{'、'.join(f.name for f in note_files)}）"
                    add_display("user", disp)
                    _notify_unreadable_files(failed_files)
                    request_resume(f"approved: {combined}" if combined else "approved")
    with c2:
        if st.button("✅　この方針で調査を開始する", type="primary", use_container_width=True):
            add_display("user", "調査方針を承認しました。調査を開始してください。")
            request_resume("approved")


# ─────────────────────────────────────────
# 結果レビュー UI
# ─────────────────────────────────────────
def render_results_detail(idata: dict):
    """調査結果の詳細（サマリー・件数・法令カード）を描画する。
    ステップ5（結果確認）とステップ6（レポート作成）の両方で再利用する。"""
    PRIORITY_STYLE = {
        "required": ("🔴", "必須対応", "#FFEBEE", "#C62828"),
        "check":    ("🟡", "要確認",   "#FFFDE7", "#F57F17"),
    }

    st.write(idata.get("summary", ""))

    rc = idata.get("risk_count", {})
    c1, c2 = st.columns(2)
    c1.metric("🔴 必須対応", rc.get("required", 0), help="稼働前に必ず届出・対応が必要な法令数")
    c2.metric("🟡 要確認",   rc.get("check", 0),    help="仕様確定後に判断が必要な法令数")

    # 網羅性チェックの結果（論点ごとの✅/⚠️判定を明示。OKの場合もOKと表示する）
    issues_list = idata.get("issues") or []
    uncovered = idata.get("uncovered_issues") or []
    issue_coverage = idata.get("issue_coverage") or {}
    coverage_failed = idata.get("coverage_check_failed") or False
    web_unconfirmed = idata.get("web_search_unconfirmed") or False
    covered_n = len([i for i in issues_list if i not in uncovered])

    if web_unconfirmed:
        st.warning(
            "🌐 Web情報（条例・届出先など）は未確認です。この調査ではWeb検索が"
            "1件も実行・成功できませんでした（GEMINI_API_KEY未設定、または継続的な"
            "エラー）。e-Gov法令API・社内文書に収載のない条例・届出先の情報は"
            "反映されていない可能性があります。"
        )

    if coverage_failed:
        st.warning(
            "🧮 網羅性チェックを実行できませんでした（AI呼び出しエラー）。"
            "論点のカバー状況は自動確認されていないため、調査論点への対応状況を"
            "手動でご確認ください。"
        )
    elif issues_list and not uncovered:
        st.success(
            f"🧮 網羅性チェック OK：調査論点 {len(issues_list)}件すべてについて、"
            f"対応する法令・情報の収集を確認しました。"
        )
    if uncovered:
        summary_line = (
            f'網羅性チェック：論点 {len(issues_list)}件中 {covered_n}件はカバー済み。'
            f'下記 {len(uncovered)}件は対応する法令・情報を確認できませんでした。'
            if issues_list else
            'AI調査（e-Gov・Web検索）では下記の論点に対応する法令・情報を確認できませんでした。'
        )
        items_html = "".join(
            f'<li style="margin-bottom:6px;line-height:1.6;">{html_lib.escape(to_ja_field_names(u))}</li>'
            for u in uncovered
        )
        st.markdown(
            f'<div style="border:1px solid #EF9A9A;border-left:6px solid #C62828;'
            f'border-radius:8px;background:#FFEBEE;padding:14px 18px;margin:10px 0;">'
            f'<div style="color:#B71C1C;font-weight:700;font-size:16px;margin-bottom:6px;">'
            f'🚨 網羅性チェック NG：対応法令を確認できなかった論点</div>'
            f'<div style="color:#7F1D1D;font-size:16px;margin-bottom:10px;">'
            f'{summary_line} 確認漏れ・届出漏れを防ぐため、担当部署・所轄機関への直接確認を推奨します。</div>'
            f'<ul style="margin:0;padding-left:20px;color:#4E342E;font-size:16px;">{items_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if issues_list:
        with st.expander(
            f"🧮 網羅性チェックの詳細（論点 {len(issues_list)}件：✅ カバー済み {covered_n} ／ "
            f"⚠️ 未カバー {len(issues_list) - covered_n}）"
        ):
            st.caption(
                "調査開始前に洗い出した論点ごとに、対応する法令・情報が収集できたかを"
                "AIの調査完了判断とは独立にチェックした結果です。"
            )
            for i in issues_list:
                if i in uncovered:
                    st.markdown(
                        f'<div style="font-size:13px;color:#B71C1C;margin:3px 0;">'
                        f'⚠️ {html_lib.escape(to_ja_field_names(i))}'
                        f'　<b>― 対応情報なし（手動確認を推奨）</b></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    covered_by = issue_coverage.get(i) or []
                    covered_by_html = (
                        f'<div style="font-size:11px;color:#558B2F;margin:1px 0 4px 1.6em;">'
                        f'└ カバー元：{html_lib.escape("、".join(covered_by))}</div>'
                        if covered_by else ""
                    )
                    st.markdown(
                        f'<div style="font-size:13px;color:#2E7D32;margin:3px 0;">'
                        f'✅ {html_lib.escape(to_ja_field_names(i))}</div>{covered_by_html}',
                        unsafe_allow_html=True,
                    )

    st.divider()
    law_items = idata.get("law_items", [])

    # search_results から law_name → law_id の逆引きマップを構築
    _law_id_map: dict[str, str] = {}
    _search_results = get_state_value(st.session_state.thread_id, "search_results") or []
    for _r in _search_results:
        _t = _r.get("title", "")
        _id = _r.get("law_id", "")
        if _t and _id and _t not in _law_id_map:
            _law_id_map[_t] = _id

    # 全法令の条文を一括取得（未キャッシュ分のみ）
    # search_results から law_revision_id の逆引きマップも構築
    _law_rev_id_map: dict[str, str] = {}
    for _r in _search_results:
        _t = _r.get("title", "")
        _rid = _r.get("law_revision_id", "")
        if _t and _rid and _t not in _law_rev_id_map:
            _law_rev_id_map[_t] = _rid

    # 条番号単位で取得済みを管理し、再調査で条番号が増えた場合も差分だけ取得する
    _pending_fetches = []
    for _law in law_items:
        _lid = _law.get("law_id", "") or _law_id_map.get(_law.get("law_name", ""), "")
        _rid = _law.get("law_revision_id", "") or _law_rev_id_map.get(_law.get("law_name", ""), "")
        _arts = [a for a in _law.get("relevant_articles", []) if a and a.strip()]
        _cache_k = f"article_text_{_lid}"
        _fetched_k = f"article_fetched_{_lid}"
        _fetched: set = st.session_state.get(_fetched_k, set())
        _missing = [a for a in _arts if a not in _fetched]
        if _lid and _missing:
            _pending_fetches.append((_lid, _rid, _missing, _cache_k, _fetched_k))
    if _pending_fetches:
        with st.spinner(f"条文を取得中...（{len(_pending_fetches)}件）"):
            for _lid, _rid, _arts, _ck, _fk in _pending_fetches:
                got = fetch_article_text(_lid, _arts, _rid)
                merged = dict(st.session_state.get(_ck, {}))
                merged.update(got)
                st.session_state[_ck] = merged
                st.session_state[_fk] = set(st.session_state.get(_fk, set())) | set(_arts)
                # 条見出し（例：（建築確認））も取得してキャッシュ（XMLは取得済みなので即時）
                caps = fetch_article_captions(_lid, _arts, _rid)
                merged_caps = dict(st.session_state.get(f"article_caption_{_lid}", {}))
                merged_caps.update(caps)
                st.session_state[f"article_caption_{_lid}"] = merged_caps

    for i, law in enumerate(law_items):
        p = law.get("priority", "check")
        icon, label, bg_color, border_color = PRIORITY_STYLE.get(p, PRIORITY_STYLE["check"])
        law_name = law.get("law_name", "")
        law_id = law.get("law_id", "") or _law_id_map.get(law_name, "")

        relevant_articles = [a for a in law.get("relevant_articles", []) if a and a.strip()]
        deliveries = law.get("deliveries", [])

        # 届出施設を重複なしで収集
        authorities = list(dict.fromkeys(
            d.get("authority", "") for d in deliveries if d.get("authority", "")
        ))

        with st.expander(f"{icon} {law_name}　　{label}", expanded=True):
            # 適用理由
            st.markdown(
                f'<div style="font-size:13px;color:#444;padding:6px 0 10px;">'
                f'📌 <em>{html_lib.escape(law.get("applicability", ""))}</em></div>',
                unsafe_allow_html=True,
            )

            # 条番号 + 届出施設 サマリーカード
            cache_key = f"article_text_{law_id}"
            cached_texts: dict = st.session_state.get(cache_key, {}) if law_id else {}
            caption_map: dict = st.session_state.get(f"article_caption_{law_id}", {}) if law_id else {}

            # 条番号→章タイトルのマップ（法令単位で一度だけ取得。XMLはキャッシュ済みのため通常は即時）
            chapter_cache_key = f"article_chapter_{law_id}"
            if law_id and relevant_articles and chapter_cache_key not in st.session_state:
                try:
                    st.session_state[chapter_cache_key] = fetch_article_chapters(
                        law_id,
                        law.get("law_revision_id", "") or _law_rev_id_map.get(law_name, ""),
                    )
                except Exception:
                    st.session_state[chapter_cache_key] = {}
            chapter_map: dict = st.session_state.get(chapter_cache_key, {}) if law_id else {}

            def _with_caption(a: str) -> str:
                # 見出しは e-Gov 取得分（本物）のみ表示する。
                # 旧データに残る LLM 由来の（）付き見出しは信頼できないため取り除く
                base = re.sub(r"（[^）]*）", "", a).strip()
                return base + caption_map.get(base, caption_map.get(a, ""))

            def _art_link(a: str) -> str:
                a = html_lib.escape(_with_caption(a))
                if law_id:
                    return (f'<a href="https://laws.e-gov.go.jp/law/{law_id}" target="_blank" '
                            f'style="color:#1565C0;text-decoration:underline;">{a}</a>')
                return f'<span style="color:#1565C0;">{a}</span>'

            def _chapter_of(a: str) -> str:
                # "第28条の2第1項（見出し）" → "第28条の2" に正規化して章を引く
                m = re.match(r"第\d+条(?:の\d+)*", re.sub(r"（[^）]*）", "", a).strip())
                return chapter_map.get(m.group(0), "") if m else ""

            # 条例・自治体系の法令か（e-Gov 未収載のため案内を出し分ける）
            is_ordinance = any(k in law_name for k in ("条例", "横浜市", "神奈川県"))

            if not relevant_articles:
                if law_id:
                    # e-Gov 収載の法令だが、適用未確定などの理由で条文を特定していない
                    art_str = (
                        '<span style="color:#888;">条番号未特定'
                        '<span style="font-size:11px;">'
                        '（この設備に明確に対応する条文を自動特定できませんでした。'
                        '適用要否が未確定の法令で発生します。適用が確定した場合は'
                        '再調査で特定できます。下の e-Gov リンクから直接確認も可能です）'
                        '</span></span>'
                    )
                elif is_ordinance:
                    art_str = (
                        '<span style="color:#888;">条番号確認中'
                        '<span style="font-size:11px;">'
                        '（横浜市・神奈川県の条例は e-Gov 未収載のため、原文照合による'
                        '条番号の自動特定ができません。下の例規集リンクから'
                        '直接ご確認ください）'
                        '</span></span>'
                    )
                else:
                    art_str = (
                        '<span style="color:#888;">条番号未特定'
                        '<span style="font-size:11px;">'
                        '（e-Gov でこの名称の法令を特定できませんでした。複数法令の'
                        '総称や名称の揺れで発生します。正式な法令名で e-Gov を'
                        '直接検索してご確認ください）'
                        '</span></span>'
                    )
            elif any(_chapter_of(a) for a in relevant_articles):
                # 条番号の昇順に並べてから章ごとにグルーピング表示
                # （条番号は章立て順に振られているため、章も昇順に並ぶ）
                _groups: dict = {}
                for a in sorted(relevant_articles, key=article_sort_key):
                    _groups.setdefault(_chapter_of(a), []).append(a)
                art_str = "".join(
                    '<div style="margin:2px 0;">'
                    + (f'<div style="font-size:12px;color:#555;font-weight:700;">【{html_lib.escape(ch)}】</div>' if ch else "")
                    + '<div style="padding-left:1em;">' + "　".join(_art_link(a) for a in arts) + "</div>"
                    + "</div>"
                    for ch, arts in _groups.items()
                )
            else:
                art_str = "　".join(_art_link(a) for a in relevant_articles)
            auth_str = "　/　".join(html_lib.escape(a) for a in authorities) \
                if authorities else '<span style="color:#888;">―</span>'

            st.markdown(
                f'<div style="border:1px solid #E0E0E0;border-radius:6px;overflow:hidden;margin-bottom:10px;">'
                f'<div style="display:flex;align-items:flex-start;background:#EFF3FF;border-bottom:1px solid #E0E0E0;">'
                f'<div style="padding:7px 12px;color:#1565C0;font-weight:700;font-size:13px;min-width:90px;white-space:nowrap;">📖 条番号</div>'
                f'<div style="padding:7px 12px;font-size:13px;">{art_str}</div>'
                f'</div>'
                f'<div style="display:flex;align-items:center;background:#F3FBF0;">'
                f'<div style="padding:7px 12px;color:#2E7D32;font-weight:700;font-size:13px;min-width:90px;white-space:nowrap;">🏛️ 届出施設</div>'
                f'<div style="padding:7px 12px;font-size:13px;color:#1B5E20;font-weight:600;">{auth_str}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # 条文インライン表示。条例は e-Gov 未収載のため、行き止まりになる
            # e-Gov 検索リンクは出さず、例規集への案内文に切り替える
            if law_id:
                egov_link_html = (
                    f'<a href="https://laws.e-gov.go.jp/law/{law_id}" target="_blank" '
                    f'style="font-size:12px;color:#1565C0;">🔗 e-Gov で条文を確認</a>'
                )
            elif is_ordinance:
                egov_link_html = (
                    f'<span style="font-size:12px;">'
                    f'{ordinance_links_html(law_name)}'
                    f'<span style="color:#888;">　（e-Gov 未収載のため公式例規集で原文をご確認ください）</span>'
                    f'</span>'
                )
            else:
                egov_link_html = (
                    f'<a href="https://laws.e-gov.go.jp/search?lawname={urllib.parse.quote(law_name)}" '
                    f'target="_blank" style="font-size:12px;color:#1565C0;">'
                    f'🔗 e-Gov で法令名を検索</a>'
                )
            # キャッシュには過去の調査ラウンドの条文も残るため、
            # 現在の relevant_articles に含まれる条番号だけを表示する
            art_pairs = [
                (a, cached_texts[a])
                for a in sorted(relevant_articles, key=article_sort_key)
                if a in cached_texts
            ]
            if art_pairs:
                # 条番号カードと同様に、章が変わる位置に章見出しを挿入する
                _rows: list = []
                _last_ch = None
                for ref, text in art_pairs:
                    _ch = _chapter_of(ref)
                    if _ch and _ch != _last_ch:
                        _rows.append(
                            f'<div style="font-size:12px;color:#555;font-weight:700;'
                            f'margin:6px 0 4px;">【{html_lib.escape(_ch)}】</div>'
                        )
                    _last_ch = _ch
                    _rows.append(
                        f'<div style="margin-bottom:10px;">'
                        f'<div style="font-size:12px;font-weight:700;color:#1565C0;margin-bottom:3px;">{html_lib.escape(_with_caption(ref))}</div>'
                        f'<div style="font-size:12px;color:#333;line-height:1.75;white-space:pre-wrap;">{html_lib.escape(text)}</div>'
                        f'</div>'
                    )
                art_rows = "".join(_rows)
                st.markdown(
                    f'<div style="background:#F8F9FA;border-left:3px solid #1565C0;'
                    f'border-radius:0 4px 4px 0;padding:10px 14px;margin-bottom:10px;">'
                    f'<div style="font-size:11px;color:#888;margin-bottom:6px;">📜 条文（e-Gov）</div>'
                    f'{art_rows}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                # 条文取得できなかった場合はリンクのみ表示（エラー文言なし）
                st.markdown(egov_link_html, unsafe_allow_html=True)

            # 届出・申請事項
            if deliveries:
                st.markdown("**📋 届出・申請事項**")
                for d in deliveries:
                    dp = d.get("priority", "check")
                    d_icon, d_label, d_bg, d_border = PRIORITY_STYLE.get(dp, PRIORITY_STYLE["check"])
                    article_ref = d.get("law_article", "")
                    if not article_ref and relevant_articles:
                        article_ref = "・".join(relevant_articles)
                    article_html = (
                        f'<span style="color:#283593;">📖 {html_lib.escape(law_name)}&nbsp;{html_lib.escape(article_ref)}</span>'
                        if article_ref else
                        f'<span style="color:#888;">📖 条文番号確認中</span>'
                    )
                    authority_val = d.get("authority", "")
                    basis = d.get("authority_basis", "")
                    src_url = d.get("authority_source_url", "")
                    src_link = (
                        f'　<a href="{html_lib.escape(src_url)}" target="_blank" '
                        f'style="color:#1565C0;">🔗 '
                        f'{html_lib.escape((d.get("authority_source_title") or "出典ページ")[:40])}</a>'
                        if src_url else ""
                    )
                    basis_html = (
                        f'<div style="margin-top:4px;font-size:11.5px;color:#607D8B;line-height:1.6;">'
                        f'└ 届出先の根拠：{html_lib.escape(basis)}{src_link}</div>'
                        if (basis or src_url) else ""
                    )
                    st.markdown(
                        f'<div style="background:{d_bg};border-left:4px solid {d_border};'
                        f'padding:9px 13px;border-radius:4px;margin:4px 0;">'
                        f'<div style="font-weight:600;font-size:14px;">{d_icon} {html_lib.escape(d.get("item", ""))}</div>'
                        f'<div style="margin-top:5px;font-size:12px;color:#555;display:flex;flex-wrap:wrap;gap:12px;">'
                        f'{article_html}'
                        f'<span style="color:#1B5E20;font-weight:600;">🏛️ {html_lib.escape(authority_val)}</span>'
                        f'<span>⏰ {html_lib.escape(d.get("deadline", ""))}</span>'
                        f'</div>'
                        f'{basis_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(
                    "📋 届出・申請事項：未特定（「届出不要」の意味ではありません。"
                    "適用が確定した場合は再調査で特定できます）"
                )

            st.markdown("")

            # 社内対応事項
            internal = law.get("internal_actions", [])
            if internal:
                st.markdown("**🏢 社内対応事項**")
                for act in internal:
                    st.markdown(
                        f'<div style="background:#F3F4F6;border-left:4px solid #6B7280;'
                        f'padding:8px 12px;border-radius:4px;margin:4px 0;">'
                        f'● <strong>{html_lib.escape(act.get("item", ""))}</strong><br>'
                        f'<span style="font-size:12px;color:#555;">⏰ 期限：{html_lib.escape(act.get("deadline", ""))}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # 確認したうえで非該当と判断した法令（判断理由をレビュー可能にし、暗黙の見落としを防ぐ）
    excluded = idata.get("excluded_laws") or []
    if excluded:
        with st.expander(
            f"🚫 確認のうえ非該当と判断した法令（{len(excluded)}件）― 判断理由に誤りがないかご確認ください",
            expanded=False,
        ):
            for e in excluded:
                st.markdown(
                    f'<div style="background:#FAFAFA;border-left:3px solid #9E9E9E;'
                    f'padding:8px 12px;border-radius:4px;margin:4px 0;font-size:13px;">'
                    f'<strong>{html_lib.escape(e.get("law_name", ""))}</strong><br>'
                    f'<span style="color:#555;">{html_lib.escape(to_ja_field_names(e.get("reason", "")))}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.caption("※ 前提となる設備情報が変わった場合は再調査してください。")

    # Google 検索の検索候補（searchEntryPoint）。
    # Gemini API 追加利用規約により、Grounding の検索結果（出典リンク等）を
    # 表示する際は検索候補もあわせて表示する必要がある。表示専用であり、
    # レポート・ケースメモリには保存しない
    suggestions = get_state_value(st.session_state.thread_id, "search_suggestions") or []
    if suggestions:
        st.markdown("**🔎 Google 検索候補**")
        st.caption(
            "Web検索（Google検索グラウンディング）に関連する Google の検索候補です。"
            "クリックすると Google 検索の結果ページが新しいタブで開きます。"
        )
        # Gemini API 追加利用規約により、検索候補の表示は最大5件までとする
        for s in suggestions[:5]:
            if s.get("html"):
                st.iframe(_suggestion_html(s["html"]), height=72)
        if len(suggestions) > 5:
            st.caption(
                f"※ 検索候補は Gemini API 利用規約に基づき最大5件まで表示しています"
                f"（他 {len(suggestions) - 5}件は非表示）。"
            )


def render_results_review(idata: dict):
    # 調査方針の内容・調査結果の詳細は、それぞれ「3. 調査方針確認」「5. 調査結果確認」
    # バナー直下（上部）に常時表示される。ここでは操作ボタンのみを置く。
    st.caption(
        "⚠️ 抜けている法令・届出施設がないかご確認ください。"
        "不足や気になる点があれば「調査不足あり・追加で調査してほしい」から再調査できます。"
    )
    c1, c2 = st.columns(2)
    with c1:
        if "show_reinvest_form" not in st.session_state:
            st.session_state.show_reinvest_form = False
        if st.button("🔍　調査不足あり・追加で調査してほしい", type="primary", use_container_width=True, key="toggle_reinvest"):
            st.session_state.show_reinvest_form = not st.session_state.show_reinvest_form
            st.rerun()
        if st.session_state.get("show_reinvest_form"):
            with st.form("results_reinvestigate", clear_on_submit=True):
                req = st.text_area(
                    "追加調査の依頼内容：",
                    placeholder="例：高圧ガス保安法の確認が抜けていそう。冷媒の充填量の観点でも再調査してください。",
                )
                reinvest_files = st.file_uploader(
                    "📎 追加資料（任意）。内容を読み取り、依頼内容と一緒に再調査の参考情報としてAIに渡します",
                    type=["pdf", "docx", "xlsx", "xlsm", "pptx", "txt"],
                    accept_multiple_files=True,
                    key="reinvest_files",
                )
                if st.form_submit_button("🔍　この内容で追加調査を依頼する", type="primary", use_container_width=True):
                    if req.strip() or reinvest_files:
                        st.session_state.show_reinvest_form = False
                        with st.spinner("添付資料を読み取り中..."):
                            doc_note, failed_files = _build_attachment_note(reinvest_files)
                        disp = f"追加調査を依頼しました：{req.strip()}"
                        if reinvest_files:
                            disp += f"（添付資料：{'、'.join(f.name for f in reinvest_files)}）"
                        add_display("user", disp)
                        _notify_unreadable_files(failed_files)
                        request_resume(f"reinvestigate: {(req.strip() + doc_note).strip()}")
                    else:
                        st.warning("依頼内容を入力するか、資料を添付してください。")
    with c2:
        if st.button("📝　調査不足なし・レポートを作成する", type="primary", use_container_width=True):
            add_display("user", "結果レビューが完了しました。レポートを作成してください。")
            request_resume({"decisions": {}})


# ─────────────────────────────────────────
# レポートレビュー UI
# ─────────────────────────────────────────
def render_report_review(idata: dict):
    # 調査方針の内容・調査結果の詳細は、それぞれのバナー直下（上部）に常時
    # 表示されるため、ここでは再掲しない（バナー二重表示・重複描画を避ける）。
    st.info("レポートの生成が完了しました。内容をご確認のうえ承認してください。")

    report_html = idata.get("report_html", "")
    case_id     = idata.get("case_id", "")

    if report_html:
        from datetime import datetime
        file_name = datetime.now().strftime("%Y%m%d_%H%M") + "_法令･届出施設確認サポートAI作成レポート.html"
        st.download_button(
            label     = "⬇️ レポートをダウンロード",
            data      = report_html.encode("utf-8"),
            file_name = file_name,
            mime      = "text/html",
            use_container_width=True,
            type      = "primary",
        )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if "show_revise_form" not in st.session_state:
            st.session_state.show_revise_form = False
        if st.button("✏️　修正・追加調査を依頼する", type="primary", use_container_width=True, key="toggle_revise"):
            st.session_state.show_revise_form = not st.session_state.show_revise_form
            st.rerun()
        if st.session_state.get("show_revise_form"):
            with st.form("report_revise", clear_on_submit=True):
                mode = st.radio(
                    "依頼の種類",
                    ["📝 文面・表現を修正したい", "🔍 内容が足りない・再調査したい"],
                    help="「文面修正」はAIが表現を直して再生成します。"
                         "「再調査」は不足分をe-Gov/Web検索からやり直します。",
                )
                txt = st.text_area(
                    "依頼内容：",
                    placeholder="文面修正の例：○○の対応内容をもっと具体的な表現にしてほしい\n"
                                "再調査の例：消防法の危険物の届出が抜けていそうなので再確認して",
                )
                revise_files = st.file_uploader(
                    "📎 追加資料（任意）。内容を読み取り、依頼内容と一緒にAIに渡します",
                    type=["pdf", "docx", "xlsx", "xlsm", "pptx", "txt"],
                    accept_multiple_files=True,
                    key="report_revise_files",
                )
                if st.form_submit_button("この内容で依頼する", type="primary", use_container_width=True):
                    if not (txt.strip() or revise_files):
                        st.warning("依頼内容を入力するか、資料を添付してください。")
                    else:
                        st.session_state.show_revise_form = False
                        with st.spinner("添付資料を読み取り中..."):
                            doc_note, failed_files = _build_attachment_note(revise_files)
                        combined = (txt.strip() + doc_note).strip()
                        names = (
                            f"（添付資料：{'、'.join(f.name for f in revise_files)}）"
                            if revise_files else ""
                        )
                        if mode.startswith("📝"):
                            add_display("user", f"レポート文面の修正を依頼：{txt.strip()}{names}")
                            _notify_unreadable_files(failed_files)
                            request_resume(f"refine: {combined}")
                        else:
                            add_display("user", f"追加調査を依頼：{txt.strip()}{names}")
                            _notify_unreadable_files(failed_files)
                            request_resume(f"reinvestigate: {combined}")
    with col2:
        if st.button("✅　確認完了・承認する", type="primary", use_container_width=True):
            st.session_state.report_html = report_html  # complete画面でも参照できるよう保存
            add_display("user", "レポートを確認しました。承認します。")
            request_resume("approved")


# ─────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # この後の render_input で調査・レポート生成などの重い処理が実行される
        # サイクルでは、ボタン操作が Streamlit の rerun を誘発して処理を中断
        # してしまうため、サイドバーの操作を無効化する
        busy = "pending_resume_decision" in st.session_state
        # AI利用コスト（スクロールせずに見えるよう最上部に表示。詳細は折りたたみ）
        # Langfuse 設定時：Langfuse が計算した実測コストを取得して表示する
        # （アプリ側では概算しない。表示は数秒〜十数秒遅れることがある）
        if langfuse_enabled():
            jpy_rate = float(os.getenv("LLM_COST_JPY_RATE", "150"))
            # Langfuse 側の集計反映は数十秒遅れるため、結果はキャッシュしつつ
            # 自動で再取得する（未反映のうちは15秒間隔・反映後は60秒間隔）。
            if st.session_state.ui_phase != "start" and not busy:
                lf_cost = st.session_state.get("lf_cost")
                interval = 60 if (lf_cost and lf_cost.get("traces")) else 15
                if time.time() - st.session_state.get("lf_cost_ts", 0) > interval:
                    fetched = get_session_cost(st.session_state.thread_id)
                    if fetched is not None:
                        st.session_state.lf_cost = fetched
                    st.session_state.lf_cost_ts = time.time()
            lf_cost = st.session_state.get("lf_cost")
            if lf_cost and lf_cost.get("traces"):
                headline = (
                    f'💰 <b>AI利用コスト ${lf_cost["cost"]:.3f}</b>'
                    f'（約 {lf_cost["cost"] * jpy_rate:,.0f}円）<br>'
                    f'<span style="font-size:11px;color:#777;">'
                    f'Langfuse 実測値（ヒアリング・調査などの処理 {lf_cost["traces"]}回分の合計。'
                    f'検索回数とは別の数字です）</span>'
                )
            elif st.session_state.ui_phase == "start":
                headline = (
                    f'💰 <b>AI利用コスト</b><br>'
                    f'<span style="font-size:11px;color:#777;">'
                    f'開始後に Langfuse の実測値を表示します</span>'
                )
            else:
                headline = (
                    f'💰 <b>AI利用コスト：集計待ち</b><br>'
                    f'<span style="font-size:11px;color:#777;">'
                    f'Langfuse への反映に数十秒かかることがあります。'
                    f'画面操作時に自動更新されます</span>'
                )
            st.markdown(
                f'<div class="sidebar-card">{headline}</div>',
                unsafe_allow_html=True,
            )
            if st.session_state.ui_phase != "start":
                if st.button(
                    "🔄 コスト表示を更新", use_container_width=True, key="lf_cost_refresh",
                    disabled=busy,
                    help="調査などの処理中は、中断を防ぐため更新できません" if busy else None,
                ):
                    with st.spinner("Langfuse から取得中..."):
                        st.session_state.lf_cost = get_session_cost(st.session_state.thread_id)
                    st.rerun()
            st.divider()

        usage = st.session_state.get("llm_usage")
        if usage and usage.get("calls"):
            jpy_rate = float(os.getenv("LLM_COST_JPY_RATE", "150"))
            web = st.session_state.get("web_usage") or {}
            total_cost = usage["cost"] + (web.get("cost") or 0.0)
            has_unpriced = bool(usage.get("unpriced")) or bool(web.get("unpriced"))
            web_line = f"　Web検索: {web['requests']}回" if web.get("requests") else ""
            # 単価不明の使用分がある場合は、誤った金額を出さずその旨を明示する
            if has_unpriced and total_cost <= 0:
                headline = "💰 <b>AI利用コスト：単価不明のため金額なし</b>"
            elif has_unpriced:
                headline = (
                    f'💰 <b>AI利用コスト ${total_cost:.3f}＋α</b>'
                    f'（約 {total_cost * jpy_rate:,.0f}円＋α・一部単価不明）'
                )
            else:
                headline = (
                    f'💰 <b>AI利用コスト ${total_cost:.3f}</b>'
                    f'（約 {total_cost * jpy_rate:,.0f}円）'
                )
            st.markdown(
                f'<div class="sidebar-card">'
                f'{headline}<br>'
                f'<span style="font-size:11px;color:#777;">'
                f'LLM: {usage["calls"]}回{web_line}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("コスト内訳"):
                llm_amount = f"　${usage['cost']:.3f}" if usage["cost"] else "　（単価不明）"
                st.caption(
                    f"LLM 入力: {usage['prompt']:,} / 出力: {usage['completion']:,} トークン"
                    f"{llm_amount}"
                )
                if web.get("requests"):
                    web_amount = f"　${web['cost']:.3f}" if web.get("cost") else "　（単価不明）"
                    st.caption(
                        f"Web検索 入力: {web['prompt']:,} / 出力: {web['completion']:,} トークン"
                        f"{web_amount}"
                    )
                if has_unpriced:
                    st.caption(
                        "⚠️ 使用モデルの単価が不明な呼び出しがあり、その分は金額に含まれていません。"
                        "正確に把握するには .env で単価（LLM_COST_INPUT_PER_1M / "
                        "LLM_COST_OUTPUT_PER_1M、Web検索は GEMINI_COST_*）を設定するか、"
                        "Langfuse をご利用ください。"
                    )
                st.caption("※ 埋め込み（社内文書登録・検索）のコストは含みません。")
            st.divider()

        st.markdown("## ⚖️ ステップ")

        idata = st.session_state.interrupt_data
        # step_idx を一次情報源にする（処理中の移行も即座に反映される）。
        # step_idx は各フェーズで PHASE_INDEX と一致するよう更新されている。
        phase_idx = st.session_state.step_idx

        st.progress((phase_idx + 1) / len(PHASES))
        st.caption(f"ステップ {phase_idx + 1} / {len(PHASES)}")

        all_done = st.session_state.ui_phase == "complete"
        # バナーが表示済みのステップはクリックでその位置へスクロールできるリンクにする
        reached_steps = {
            m.get("step") for m in st.session_state.display_messages
            if m.get("role") == "step_banner"
        }
        for i, phase in enumerate(PHASES):
            if i < phase_idx or all_done:
                mark, weight = "✅", ""
            elif i == phase_idx:
                mark, weight = "▶️", "font-weight:700;"
            else:
                mark, weight = "⬜", ""
            if i in reached_steps:
                st.markdown(
                    f'<a href="#step-anchor-{i}" target="_self" '
                    f'style="display:block;text-decoration:none;color:inherit;'
                    f'{weight}margin:4px 0;">{mark} {phase}</a>',
                    unsafe_allow_html=True,
                )
            else:
                st.write(f"{mark} {phase}")
        if reached_steps:
            st.caption("クリックすると、その工程の位置までスクロールします。")

        st.divider()

        # 設備情報サマリー
        eq_info = {}
        if idata:
            eq_info = idata.get("equipment_info", {})
        if not eq_info:
            eq_info = get_state_value(st.session_state.thread_id, "equipment_info") or {}

        if eq_info:
            st.markdown("**🏭 設備情報**")
            for key, (icon, label) in EQUIPMENT_LABELS.items():
                val = eq_info.get(key, "")
                if not val:
                    continue
                sval = str(val)
                # 「ありません」等の否定表現に「あり」が部分一致して⚠️に
                # ならないよう、否定語を除外してから判定する
                if re.search(r"あり(?!ません)", sval):
                    badge = "⚠️"
                elif sval.startswith("なし") or "ありません" in sval:
                    badge = "✅"
                elif sval in ("不明", "未定", "確認中"):
                    badge = "❓"
                else:
                    badge = "📌"
                st.markdown(
                    f'<div class="sidebar-card">{badge} {icon} <b>{label}</b><br>{html_lib.escape(str(val))}</div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # 社内文書の登録・管理（Agentic RAG 検索の対象）
        with st.expander("📁 社内文書（Agentic RAG検索）"):
            del_err = st.session_state.pop("doc_delete_error", None)
            if del_err:
                st.error(
                    f"「{del_err}」の削除に失敗しました。検索インデックスから"
                    "削除できなかったため、登録状態を維持しています。"
                )
            flash = st.session_state.pop("ingest_result", None)
            if flash:
                if flash.get("added"):
                    st.success("登録完了: " + "、".join(flash["added"]))
                if flash.get("skipped"):
                    st.info("登録済みのためスキップ: " + "、".join(flash["skipped"]))
                if flash.get("failed"):
                    st.warning("テキスト抽出不可: " + "、".join(flash["failed"]))
                if flash.get("truncated"):
                    st.info(
                        "サイズ上限（5万字）を超えたため先頭のみ登録: "
                        + "、".join(flash["truncated"])
                    )
                if flash.get("error"):
                    st.error("登録に失敗しました。時間をおいて再度お試しください。")

            registered = list_internal_docs()
            if registered:
                st.caption(
                    f"登録済み {len(registered)} ファイル（調査時にAIが検索します）。"
                    "各ファイル右の 🗑️ で個別に削除できます。"
                )
                # 同名で内容の異なる資料が複数登録され得るため、doc_id を
                # ウィジェットキー・削除キーに使う（登録日時も表示して区別する）
                for r in registered:
                    c1, c2 = st.columns([5, 1])
                    c1.markdown(
                        f'<div style="font-size:12px;padding-top:6px;">📄 {html_lib.escape(r["name"])}'
                        f'<span style="color:#888;">（{r["chunks"]}チャンク'
                        + (f'／{html_lib.escape(r["registered_at"])}' if r.get("registered_at") else "")
                        + '）</span></div>',
                        unsafe_allow_html=True,
                    )
                    if c2.button("🗑️", key=f"del_doc_{r['doc_id']}", help="この文書をインデックスから削除",
                                 disabled=busy):
                        if not delete_internal_doc(r["doc_id"]):
                            st.session_state.doc_delete_error = r["name"]
                        st.rerun()

                # 全削除（誤操作防止のため確認ステップを挟む）
                if st.session_state.get("confirm_delete_all_docs"):
                    st.warning(f"登録済み {len(registered)} ファイルをすべて削除します。よろしいですか？")
                    dc1, dc2 = st.columns(2)
                    if dc1.button("削除する", type="primary", key="do_delete_all_docs",
                                  use_container_width=True, disabled=busy):
                        delete_all_internal_docs()
                        st.session_state.confirm_delete_all_docs = False
                        st.rerun()
                    if dc2.button("キャンセル", key="cancel_delete_all_docs",
                                  use_container_width=True, disabled=busy):
                        st.session_state.confirm_delete_all_docs = False
                        st.rerun()
                elif st.button("🗑️ すべて削除", key="delete_all_docs_btn", use_container_width=True,
                               disabled=busy):
                    st.session_state.confirm_delete_all_docs = True
                    st.rerun()
            else:
                st.caption(
                    "未登録です。社内規定・基準・過去の届出事例などを登録すると、"
                    "調査時にAIが社内文書も検索します。"
                )

            up_files = st.file_uploader(
                "社内文書を追加",
                type=["pdf", "docx", "xlsx", "xlsm", "pptx", "txt"],
                accept_multiple_files=True,
                key="internal_docs_uploader",
                disabled=busy,
            )
            if up_files and st.button("📥 登録（ベクトル化）", use_container_width=True,
                                      disabled=busy):
                with st.spinner("社内文書を登録中...（埋め込みを生成しています）"):
                    try:
                        st.session_state.ingest_result = ingest_internal_docs(
                            [(f.name, f.getvalue()) for f in up_files]
                        )
                    except Exception:
                        logger.error("社内文書の登録でエラー:\n%s", traceback.format_exc())
                        st.session_state.ingest_result = {"error": True}
                st.rerun()

        # ケースメモリ（承認済み過去案件・CBR）の管理
        with st.expander("📚 過去の調査事例（AIが自動で参考にします）"):
            saved_cases = list_cases()
            if saved_cases:
                st.caption(
                    f"承認済み案件 {len(saved_cases)}件。"
                    "新しい案件の分析時に、類似案件が自動で参照されます。"
                )
                for c in saved_cases:
                    cc1, cc2 = st.columns([5, 1])
                    cc1.markdown(
                        f'<div style="font-size:12px;padding-top:6px;">📋 {html_lib.escape(c["case_id"])}'
                        f'<br><span style="color:#888;">{html_lib.escape(c["equipment_type"] or "（設備種別不明）")}'
                        f'・法令{c["law_count"]}件・{c["saved_at"]}</span></div>',
                        unsafe_allow_html=True,
                    )
                    if cc2.button("🗑️", key=f"del_case_{c['case_id']}", help="この事例を削除（AIが参照しなくなります）",
                                  disabled=busy):
                        delete_case(c["case_id"])
                        st.rerun()
            else:
                st.caption(
                    "まだ事例がありません。レポートを承認すると、その案件が自動で保存され、"
                    "次回以降の類似案件の分析で参照されます（使うほど賢くなります）。"
                )

        st.divider()

        # 案件IDはレポートと同じ case_id を表示する（確定前は thread_id で代用）
        case_id = get_state_value(st.session_state.thread_id, "case_id")
        if case_id:
            st.caption(f"案件ID: {case_id}")
        else:
            st.caption(f"案件ID: {st.session_state.thread_id[:8]}...")
        st.caption("⚠️ 本ツールの提案は参考情報です。最終判断は担当者が行ってください。")


# ─────────────────────────────────────────
# 自動スクロール（最下部へ）
# ─────────────────────────────────────────
def _embed_html(html_str: str, height: int = 1):
    """HTML（JavaScript含む）を iframe として埋め込む。height は 1 以上にする。
    st.iframe は HTML 文字列を srcdoc 埋め込み（same-origin・scripts 許可）
    するので、スクリプトから window.parent.document へのアクセスも従来どおり動く。"""
    st.iframe(html_str, height=max(1, height))


def auto_scroll(duration_ms: int = 1200):
    """再描画のたびにチャット最下部（入力欄）までゆっくり自動スクロールする。
    duration_ms: スクロールにかける時間（大きいほど遅い）。"""
    # display_messages 件数を埋め込み、内容を毎回変化させて再実行を強制する
    n = len(st.session_state.get("display_messages", []))
    _embed_html(
        f"""
        <script>
            // 描画カウンタ: {n}
            const DURATION = {duration_ms};   // スクロール所要時間(ms)
            const doc = window.parent.document;
            const selectors = [
                'section.main',
                '[data-testid="stMain"]',
                '[data-testid="stAppViewContainer"]',
                '.main',
                '.appview-container',
            ];

            // 実際にスクロール可能なコンテナを1つ特定する
            const getScroller = () => {{
                for (const sel of selectors) {{
                    for (const el of doc.querySelectorAll(sel)) {{
                        if (el.scrollHeight > el.clientHeight + 4) return el;
                    }}
                }}
                return doc.scrollingElement || doc.documentElement;
            }};

            // easeInOutQuad（最初と最後がゆっくり）
            const ease = (t) => t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t+2, 2)/2;

            const smoothScroll = () => {{
                const el = getScroller();
                const start = el.scrollTop;
                const end = el.scrollHeight - el.clientHeight;
                const dist = end - start;
                if (dist <= 0) return;
                const t0 = performance.now();
                const step = (now) => {{
                    const p = Math.min((now - t0) / DURATION, 1);
                    el.scrollTop = start + dist * ease(p);
                    if (p < 1) requestAnimationFrame(step);
                }};
                requestAnimationFrame(step);
            }};

            // 描画が落ち着いてからアニメーション開始
            setTimeout(smoothScroll, 200);
        </script>
        """,
        height=1,
    )



# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────
WELCOME_HTML = """
<div style="background:#E8F5E9;border-left:4px solid #43A047;padding:14px 18px;border-radius:4px;margin-bottom:16px">
👋 <b>ようこそ！</b> 法令・届出施設確認サポートAIです。<br>
設備情報をヒアリングし、横浜市内の会社施設で設備を導入する際に必要な法令・届出施設をAIが調査・報告します。<br>
仕様書などの関連資料をアップロードすると、AIが設備情報を自動で読み取り、ヒアリングでの手入力を減らせます。<br>
まず「開始する」を押すと、AIが設備情報のヒアリング（資料がある場合は抽出結果の確認）を開始します。<br><br>
<b>進め方：</b> 1. ヒアリング → 2. 情報整理・分析 → 3. 調査方針確認 → 4. 調査実施 → 5. 調査結果確認 → 6. レポート作成 → 7. 完了確認
</div>
<div style="background:#FFF8E1;border-left:4px solid #FB8C00;padding:12px 18px;border-radius:4px;margin-bottom:16px;color:#5D4037;">
⚠️ <b>ご利用にあたって：</b>本ツールは調査を補助する「サポートAI」であり、法令の適用有無・届出の要否を<b>断定するものではありません</b>。
AIの調査結果・提案はすべて参考情報です（生成AIの利用規約上も、人の確認を経ない法的判断の自動化はできません）。
<b>最終判断は必ず担当者が行ってください。</b>
</div>
"""


def main():
    init()
    render_sidebar()

    st.title("⚖️ 法令・届出施設確認サポートAI")

    st.divider()

    if st.session_state.ui_phase == "start":
        st.markdown(WELCOME_HTML, unsafe_allow_html=True)
    else:
        # 開始後も説明文を消さず、折りたたみで常設表示する
        with st.expander("ℹ️ このツールの使い方・注意事項", expanded=False):
            st.markdown(WELCOME_HTML, unsafe_allow_html=True)
            st.caption(
                "💡 サイドバーの「📁 社内文書」に社内規定・過去の届出事例を登録しておくと、"
                "AIが調査時に社内文書も検索します。また、レポートを承認した案件は"
                "「📚 過去の調査事例」に蓄積され、次回の類似案件で自動的に参照されます"
                "（使うほど賢くなります）。"
            )
        render_messages()

    st.divider()
    render_input()

    # ヒアリング中は新しい質問が来たら最下部まで自動スクロール
    if st.session_state.ui_phase == "hearing":
        auto_scroll()


if __name__ == "__main__":
    main()
