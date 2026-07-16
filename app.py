"""
設備導入時 法令確認・届出施設確認AI
メインアプリ - Streamlit UI
"""

import os
import re
import html as html_lib
import uuid
import sys
import logging
import traceback
import urllib.parse
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.callbacks import get_openai_callback
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
    delete_file as delete_internal_doc,
    delete_all as delete_all_internal_docs,
)
from backend.case_memory import list_cases, delete_case
from backend.tools.web_search import get_gemini_usage
from backend.observability import trace_run, langfuse_enabled

# ─────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────
st.set_page_config(
    page_title="法令確認・届出施設確認AI",
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

.phase-banner {
    background: linear-gradient(135deg, #1565C0, #1976D2);
    color: white; padding: 10px 16px; border-radius: 8px;
    font-weight: 700; font-size: 14px; margin-bottom: 12px;
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
    Google が返すリンクには target 属性がなく、components.html の iframe 内で
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
        "expected_questions": 11,      # AIが質問する残り項目数（資料確定分だけ減る）
        "api_key_ok":       bool(
            os.getenv("POC_LLM_API_KEY") or os.getenv("PROD_LLM_API_KEY")
        ),
        # Gemini Web検索コストの差分計算用（セッション開始時点の累積値）
        "web_usage_baseline": get_gemini_usage(),
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
def _record_llm_usage(cb) -> None:
    """get_openai_callback の計測結果を案件単位で積算する。
    Azure等でモデル単価が取得できず cost=0 の場合は、環境変数の単価
    （LLM_COST_INPUT_PER_1M / LLM_COST_OUTPUT_PER_1M、USD/100万トークン）で概算する。"""
    cost = cb.total_cost
    if not cost and (cb.prompt_tokens or cb.completion_tokens):
        in_rate  = float(os.getenv("LLM_COST_INPUT_PER_1M",  "2.5"))
        out_rate = float(os.getenv("LLM_COST_OUTPUT_PER_1M", "10.0"))
        cost = cb.prompt_tokens / 1e6 * in_rate + cb.completion_tokens / 1e6 * out_rate
    usage = st.session_state.setdefault(
        "llm_usage", {"prompt": 0, "completion": 0, "cost": 0.0, "calls": 0}
    )
    usage["prompt"]     += cb.prompt_tokens
    usage["completion"] += cb.completion_tokens
    usage["cost"]       += cost
    usage["calls"]      += cb.successful_requests
    # Gemini Web検索の使用量も同じタイミングで同期する
    _sync_web_search_usage()


def _sync_web_search_usage() -> None:
    """Gemini Web検索のプロセス累積使用量から、このセッション分の差分を積算する。
    単価は環境変数（GEMINI_COST_INPUT_PER_1M / GEMINI_COST_OUTPUT_PER_1M、
    USD/100万トークン。既定は gemini-2.0-flash 相当）で調整できる。
    検索1回あたりの課金（無料枠超過時）は GEMINI_COST_PER_SEARCH で指定する。"""
    current = get_gemini_usage()
    baseline = st.session_state.get("web_usage_baseline") or {
        "prompt_tokens": 0, "completion_tokens": 0, "requests": 0
    }
    d_prompt   = current["prompt_tokens"]     - baseline["prompt_tokens"]
    d_complete = current["completion_tokens"] - baseline["completion_tokens"]
    d_requests = current["requests"]          - baseline["requests"]
    if d_prompt or d_complete or d_requests:
        in_rate    = float(os.getenv("GEMINI_COST_INPUT_PER_1M",  "0.10"))
        out_rate   = float(os.getenv("GEMINI_COST_OUTPUT_PER_1M", "0.40"))
        per_search = float(os.getenv("GEMINI_COST_PER_SEARCH",    "0"))
        web = st.session_state.setdefault(
            "web_usage", {"prompt": 0, "completion": 0, "requests": 0, "cost": 0.0}
        )
        web["prompt"]     += d_prompt
        web["completion"] += d_complete
        web["requests"]   += d_requests
        web["cost"] += (
            d_prompt / 1e6 * in_rate
            + d_complete / 1e6 * out_rate
            + d_requests * per_search
        )
    st.session_state.web_usage_baseline = current


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
            with get_openai_callback() as cb, trace_run(
                "hearing", session_id=st.session_state.thread_id, input=user_text,
            ) as lf:
                try:
                    workflow.invoke(
                        {"messages": [HumanMessage(content=user_text)]},
                        config={**config, **lf},
                    )
                finally:
                    # エラー時も、そこまでに消費したトークンを計上する
                    _record_llm_usage(cb)
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
    doc_texts, failed = [], []
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
                with get_openai_callback() as cb, trace_run(
                    "doc_extraction",
                    session_id=st.session_state.thread_id,
                    input=[name for name, _t in doc_texts],
                ) as lf:
                    try:
                        extracted = extract_equipment_info(doc_texts, config=lf)
                    finally:
                        _record_llm_usage(cb)
            except Exception:
                logger.error("資料からの情報抽出でエラー:\n%s", traceback.format_exc())

    st.session_state.extract_failed_files = failed
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
            with get_openai_callback() as cb, trace_run(
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
                    _record_llm_usage(cb)
        except Exception as e:
            logger.error("調査フェーズ (workflow.stream) でエラー:\n%s", traceback.format_exc())
            status.update(label="⚠️ 調査中にエラーが発生しました", state="error", expanded=True)
            status.write(str(e))
            add_display("system", "⚠️ 調査中にエラーが発生しました。もう一度お試しください。")
            _rollback_resume_banner(current_phase)
            return
        else:
            status.update(label="✅ AI調査が完了しました", state="complete", expanded=False)
    else:
        try:
            with st.spinner("処理中..."):
                with get_openai_callback() as cb, trace_run(
                    f"resume:{current_phase or 'resume'}",
                    session_id=st.session_state.thread_id,
                    input=decision,
                ) as lf:
                    try:
                        workflow.invoke(Command(resume=decision), config={**config, **lf})
                    finally:
                        _record_llm_usage(cb)
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
            st.markdown(
                f'<div class="phase-banner">{icon}　{name}</div>',
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
            "「🧠 ケースメモリ」に蓄積され、次回の類似案件で自動的に参照されます（使うほど賢くなります）。"
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
            if submitted and txt.strip():
                submit_hearing_answer(txt.strip())

        st.caption("よく使う選択肢：")
        cols = st.columns(5)
        for i, opt in enumerate(["あり", "なし", "不明", "未定", "確認中"]):
            if cols[i].button(opt, key=f"q_{opt}", use_container_width=True):
                submit_hearing_answer(opt)

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
        if st.session_state.get("report_saved_path"):
            st.caption(f"💾 レポートは `{st.session_state.report_saved_path}` に自動保存済みです。")
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
            file_name = datetime.now().strftime("%Y%m%d_%H%M") + "_法令確認･届出施設確認AI作成レポート.html"
            c2.download_button(
                label="⬇️ レポートをダウンロード",
                data=st.session_state.report_html.encode("utf-8"),
                file_name=file_name,
                mime="text/html",
                use_container_width=True,
            )


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
            f'<div style="color:#E65100;font-weight:700;font-size:15px;margin-bottom:6px;">'
            f'⚠️ 調査前に確認した方がよい不明・未定情報</div>'
            f'<div style="color:#6D4C41;font-size:13px;margin-bottom:10px;">'
            f'下記が未確定だと必要な法令確認・届出確認が漏れる恐れがあります。'
            f'「調査前に追記する」から補足してから調査を開始することを推奨します。</div>'
            f'<ul style="margin:0;padding-left:20px;color:#4E342E;font-size:14px;">{items_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander("📌 調査方針", expanded=expanded):
        st.write(to_ja_field_names(idata.get("search_plan", "")))
        st.caption("検索キーワード（e-Gov法令API）: " + " / ".join(idata.get("search_keywords", [])))
        st.caption("🌐 AIWeb検索（自動実施） ● 横浜市・神奈川県の条例・規制（公式サイト） ● 省庁ガイドライン・FAQ（厚生労働省・消防庁・環境省・国土交通省・経済産業省）")


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
                if st.form_submit_button("✅　追記して調査を開始する", type="primary", use_container_width=True):
                    st.session_state.show_note_form = False
                    add_display("user", f"調査方針を承認しました。追記：{note}")
                    request_resume(f"approved: {note}")
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
    covered_n = len([i for i in issues_list if i not in uncovered])

    if issues_list and not uncovered:
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
            f'<div style="color:#B71C1C;font-weight:700;font-size:15px;margin-bottom:6px;">'
            f'🚨 網羅性チェック NG：対応法令を確認できなかった論点</div>'
            f'<div style="color:#7F1D1D;font-size:13px;margin-bottom:10px;">'
            f'{summary_line} 確認漏れ・届出漏れを防ぐため、担当部署・所轄機関への直接確認を推奨します。</div>'
            f'<ul style="margin:0;padding-left:20px;color:#4E342E;font-size:14px;">{items_html}</ul>'
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
                        '条番号の自動特定ができません。横浜市例規集・神奈川県例規集で'
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
                    '<span style="font-size:12px;color:#888;">'
                    '📚 条例の原文は「横浜市例規集」「神奈川県例規集」で検索して'
                    'ご確認ください（e-Gov 未収載）</span>'
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
                st.caption("届出・申請事項なし（要確認）")

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
                components.html(_suggestion_html(s["html"]), height=72, scrolling=True)
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
                if st.form_submit_button("🔍　この内容で追加調査を依頼する", type="primary", use_container_width=True):
                    if req.strip():
                        st.session_state.show_reinvest_form = False
                        add_display("user", f"追加調査を依頼しました：{req.strip()}")
                        request_resume(f"reinvestigate: {req.strip()}")
                    else:
                        st.warning("依頼内容を入力してください。")
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
        file_name = datetime.now().strftime("%Y%m%d_%H%M") + "_法令確認･届出施設確認AI作成レポート.html"
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
                if st.form_submit_button("この内容で依頼する", type="primary", use_container_width=True):
                    if not txt.strip():
                        st.warning("依頼内容を入力してください。")
                    elif mode.startswith("📝"):
                        st.session_state.show_revise_form = False
                        add_display("user", f"レポート文面の修正を依頼：{txt.strip()}")
                        request_resume(f"refine: {txt.strip()}")
                    else:
                        st.session_state.show_revise_form = False
                        add_display("user", f"追加調査を依頼：{txt.strip()}")
                        request_resume(f"reinvestigate: {txt.strip()}")
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
        # AI利用コスト（スクロールせずに見えるよう最上部に表示。詳細は折りたたみ）
        usage = st.session_state.get("llm_usage")
        if usage and usage.get("calls"):
            jpy_rate = float(os.getenv("LLM_COST_JPY_RATE", "150"))
            web = st.session_state.get("web_usage") or {}
            total_cost = usage["cost"] + (web.get("cost") or 0.0)
            web_line = f"　Web検索: {web['requests']}回" if web.get("requests") else ""
            st.markdown(
                f'<div class="sidebar-card">'
                f'💰 <b>AI利用コスト ${total_cost:.3f}</b>'
                f'（約 {total_cost * jpy_rate:,.0f}円）<br>'
                f'<span style="font-size:11px;color:#777;">'
                f'LLM: {usage["calls"]}回{web_line}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("コスト内訳"):
                st.caption(
                    f"LLM 入力: {usage['prompt']:,} / 出力: {usage['completion']:,} トークン"
                    f"　${usage['cost']:.3f}"
                )
                if web.get("requests"):
                    st.caption(
                        f"Web検索 入力: {web['prompt']:,} / 出力: {web['completion']:,} トークン"
                        f"　${web['cost']:.3f}"
                    )
                st.caption("※ 埋め込み（社内文書登録・検索）のコストは含みません。")
                if langfuse_enabled():
                    lf_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                    st.caption(
                        f"※ 表示は概算です。正確なコスト・リクエスト内容のトレースは "
                        f"[Langfuse]({lf_host}) で確認できます。"
                    )
            st.divider()

        st.markdown("## ⚖️ ステップ")

        idata = st.session_state.interrupt_data
        # step_idx を一次情報源にする（処理中の移行も即座に反映される）。
        # step_idx は各フェーズで PHASE_INDEX と一致するよう更新されている。
        phase_idx = st.session_state.step_idx

        st.progress((phase_idx + 1) / len(PHASES))
        st.caption(f"ステップ {phase_idx + 1} / {len(PHASES)}")

        all_done = st.session_state.ui_phase == "complete"
        for i, phase in enumerate(PHASES):
            if i < phase_idx or all_done:
                st.write(f"✅ {phase}")
            elif i == phase_idx:
                st.write(f"▶️ **{phase}**")
            else:
                st.write(f"⬜ {phase}")

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
                if "あり" in str(val):
                    badge = "⚠️"
                elif str(val) in ("なし", "なし（確認済み）"):
                    badge = "✅"
                elif str(val) in ("不明", "未定", "確認中"):
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
                    st.error(f"登録に失敗しました: {flash['error']}")

            registered = list_internal_docs()
            if registered:
                st.caption(
                    f"登録済み {len(registered)} ファイル（調査時にAIが検索します）。"
                    "各ファイル右の 🗑️ で個別に削除できます。"
                )
                for r in registered:
                    c1, c2 = st.columns([5, 1])
                    c1.markdown(
                        f'<div style="font-size:12px;padding-top:6px;">📄 {html_lib.escape(r["name"])}'
                        f'<span style="color:#888;">（{r["chunks"]}チャンク）</span></div>',
                        unsafe_allow_html=True,
                    )
                    if c2.button("🗑️", key=f"del_doc_{r['name']}", help="この文書をインデックスから削除"):
                        delete_internal_doc(r["name"])
                        st.rerun()

                # 全削除（誤操作防止のため確認ステップを挟む）
                if st.session_state.get("confirm_delete_all_docs"):
                    st.warning(f"登録済み {len(registered)} ファイルをすべて削除します。よろしいですか？")
                    dc1, dc2 = st.columns(2)
                    if dc1.button("削除する", type="primary", key="do_delete_all_docs",
                                  use_container_width=True):
                        delete_all_internal_docs()
                        st.session_state.confirm_delete_all_docs = False
                        st.rerun()
                    if dc2.button("キャンセル", key="cancel_delete_all_docs",
                                  use_container_width=True):
                        st.session_state.confirm_delete_all_docs = False
                        st.rerun()
                elif st.button("🗑️ すべて削除", key="delete_all_docs_btn", use_container_width=True):
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
            )
            if up_files and st.button("📥 登録（ベクトル化）", use_container_width=True):
                with st.spinner("社内文書を登録中...（埋め込みを生成しています）"):
                    try:
                        st.session_state.ingest_result = ingest_internal_docs(
                            [(f.name, f.getvalue()) for f in up_files]
                        )
                    except Exception as e:
                        logger.error("社内文書の登録でエラー:\n%s", traceback.format_exc())
                        st.session_state.ingest_result = {"error": str(e)}
                st.rerun()

        # ケースメモリ（承認済み過去案件・CBR）の管理
        with st.expander("🧠 ケースメモリ（過去案件）"):
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
                    if cc2.button("🗑️", key=f"del_case_{c['case_id']}", help="この事例をケースメモリから削除"):
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
    """HTML（JavaScript含む）を iframe として埋め込む。height は 1 以上にする。"""
    components.html(html_str, height=max(1, height))


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
def main():
    init()
    render_sidebar()

    st.title("⚖️ 法令確認・届出施設確認AI")
    st.caption(
        "設備導入時の法令確認・届出施設確認をAIがEnd to Endでサポートします。"
        "　|　対象：横浜市内の会社施設"
    )

    st.divider()

    if not st.session_state.display_messages and st.session_state.ui_phase == "start":
        st.markdown("""
<div style="background:#E8F5E9;border-left:4px solid #43A047;padding:14px 18px;border-radius:4px;margin-bottom:16px">
👋 <b>ようこそ！</b> 法令確認・届出施設確認AIです。<br>
設備情報をヒアリングし、横浜市内の会社施設で設備を導入する際に必要な法令・届出施設をAIが調査・報告します。<br>
仕様書などの関連資料をアップロードすると、AIが設備情報を自動で読み取り、ヒアリングでの手入力を減らせます。<br>
まず「開始する」を押すと、AIが設備情報のヒアリング（資料がある場合は抽出結果の確認）を開始します。<br><br>
<b>進め方：</b> 1. ヒアリング → 2. 情報整理・分析 → 3. 調査方針確認 → 4. 調査実施 → 5. 調査結果確認 → 6. レポート作成 → 7. 完了確認
</div>
""", unsafe_allow_html=True)
    else:
        render_messages()

    st.divider()
    render_input()

    # ヒアリング中は新しい質問が来たら最下部まで自動スクロール
    if st.session_state.ui_phase == "hearing":
        auto_scroll()


if __name__ == "__main__":
    main()
