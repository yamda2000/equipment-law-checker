"""
設備導入時 法令確認・届出施設確認AI
メインアプリ - Streamlit UI
"""

import os
import re
import uuid
import sys
import urllib.parse
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

from backend.workflow import (
    workflow, get_interrupt_data, get_all_messages, get_state_value
)
from backend.tools.egov import fetch_article_text

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
    background: #F3E5F5;
    border-left: 4px solid #7B1FA2;
    border-radius: 0 12px 12px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px; color: #4A148C;
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
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 定数
# ─────────────────────────────────────────
PHASES = ["1. ヒアリング", "2. 情報整理", "3. 調査方針確認", "4. 調査実施", "5. 結果確認", "6. レポート作成", "7. 完了"]

PHASE_ICONS = ["💬", "📊", "📋", "🔍", "✅", "📄", "🎉"]

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
    "fire_exhaust":       ("🔥", "火気・排気"),
    "wastewater":         ("💧", "排水"),
    "noise_vibration":    ("📢", "騒音・振動"),
    "radiation":          ("☢️", "放射線・X線"),
    "construction":       ("🏗️", "建屋改修"),
}



# ─────────────────────────────────────────
# セッション初期化
# ─────────────────────────────────────────
def init():
    defaults = {
        "thread_id":        str(uuid.uuid4()),
        "ui_phase":         "start",   # start | hearing | interrupt | complete
        "interrupt_data":   None,
        "step_idx":         0,         # 現在のステップ番号（0〜6）
        "review_decisions": {},
        "report_html":      "",
        "display_messages": [],
        "msg_count":        0,         # 表示済みメッセージ数（重複防止）
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
    if msgs and msgs[-1].get("role") == "step_banner" and msgs[-1].get("step") == step_idx:
        return
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
        elif isinstance(m, ToolMessage):
            add_step_banner(1)
            add_display("system", "AIがヒアリング情報を分析し、調査方針を作成しました。調査方針をご確認ください。")
            st.session_state.step_idx = 1   # ステップ2: 情報整理
    st.session_state.msg_count = len(all_msgs)


# ─────────────────────────────────────────
# LangGraph 呼び出し（ヒアリング中）
# ─────────────────────────────────────────
def invoke_hearing(user_text: str):
    old_count = st.session_state.msg_count
    config = get_config()

    # AIの質問が11件揃っていれば次のinvokeでヒアリング完了→情報整理に移行する
    ai_count = sum(1 for m in st.session_state.display_messages if m.get("role") == "ai")
    hearing_ending = ai_count >= 11
    if hearing_ending:
        idx, icon, name = 1, PHASE_ICONS[1], PHASES[1]
        st.markdown(
            f'<div class="phase-banner">{icon}　{name}</div>',
            unsafe_allow_html=True,
        )

    with st.spinner("情報整理中..." if hearing_ending else "AIが考えています..."):
        workflow.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )

    _process_new_msgs(old_count, skip_human=True)

    idata = get_interrupt_data(st.session_state.thread_id)
    if idata:
        st.session_state.interrupt_data = idata
        st.session_state.ui_phase = "interrupt"
        st.session_state.step_idx = PHASE_INDEX.get(idata.get("phase", ""), 1)
        add_step_banner(st.session_state.step_idx)
    # hearing 継続中はステップ変更なし


# ─────────────────────────────────────────
# LangGraph 呼び出し（interrupt 再開）
# ─────────────────────────────────────────
def resume_graph(decision):
    old_count = st.session_state.msg_count
    config = get_config()

    # 現フェーズに応じて「実行中」バナーを先行表示
    current_phase = (st.session_state.interrupt_data or {}).get("phase", "")
    if current_phase == "policy_review":
        add_step_banner(3)  # ステップ4: 調査実施
    elif current_phase == "results_review":
        add_step_banner(5)  # ステップ6: レポート作成

    if current_phase == "policy_review":
        # 調査フェーズは各ステップの進捗をライブ表示（stream_mode="custom"）
        status = st.status("🔎 AI調査を開始しています...", expanded=True)
        try:
            for chunk in workflow.stream(
                Command(resume=decision), config=config, stream_mode="custom"
            ):
                entry = chunk.get("progress") if isinstance(chunk, dict) else None
                if entry:
                    status.update(label=f"AI調査中... {entry}")
                    status.write(entry)
        except Exception as e:
            status.update(label="⚠️ 調査中にエラーが発生しました", state="error", expanded=True)
            status.write(str(e))
            raise
        else:
            status.update(label="✅ AI調査が完了しました", state="complete", expanded=False)
    else:
        with st.spinner("処理中..."):
            workflow.invoke(Command(resume=decision), config=config)

    _process_new_msgs(old_count, skip_human=False)

    idata = get_interrupt_data(st.session_state.thread_id)
    if idata:
        st.session_state.interrupt_data = idata
        st.session_state.ui_phase = "interrupt"
        st.session_state.step_idx = PHASE_INDEX.get(idata.get("phase", ""), 0)
        add_step_banner(st.session_state.step_idx)
    else:
        rhtml = get_state_value(st.session_state.thread_id, "report_html")
        if rhtml:
            st.session_state.report_html = rhtml
        st.session_state.ui_phase = "complete"
        st.session_state.interrupt_data = None
        st.session_state.step_idx = 6
        add_step_banner(6)


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
    for msg in st.session_state.display_messages:
        role = msg["role"]

        if role == "step_banner":
            idx  = msg["step"]
            icon = PHASE_ICONS[idx]
            name = PHASES[idx]
            st.markdown(
                f'<div class="phase-banner">{icon}　{name}</div>',
                unsafe_allow_html=True,
            )
            continue

        if role == "search_log":
            st.markdown(
                f'<div class="search-log-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            continue

        content = msg["content"].replace("\n", "<br>")
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

    # ── 開始前 ──
    if ui == "start":
        if not st.session_state.api_key_ok:
            mode = os.getenv("LLM_MODE", "poc")
            key_name = "POC_LLM_API_KEY" if mode != "prod" else "PROD_LLM_API_KEY"
            st.error(f"⚠️ {key_name} が設定されていません。`.env` ファイルを作成してAPIキーを設定してください。")
            st.code("copy .env.example .env\n# .env を編集して LLM_MODE と API キーを設定")
            return
        st.info("👇 「開始する」を押すと、AIが設備情報のヒアリングを開始します。")
        if st.button("▶️　法令確認・届出施設確認を開始する", type="primary", use_container_width=True):
            add_step_banner(0)
            add_display("user", "法令確認・届出施設確認を開始します。")
            st.session_state.step_idx = 0
            invoke_hearing("法令確認・届出施設確認を開始します。設備情報のヒアリングをお願いします。")
            st.session_state.ui_phase = "hearing"
            st.rerun()

    # ── ヒアリング中 ──
    elif ui == "hearing":
        with st.form("hearing_form", clear_on_submit=True):
            txt = st.text_input(
                "AIへの回答を入力してください",
                placeholder="こちらに記入してください。",
            )
            submitted = st.form_submit_button("送信 ➤", type="primary", use_container_width=True)
            if submitted and txt.strip():
                add_display("user", txt.strip())
                invoke_hearing(txt.strip())
                st.rerun()

        st.caption("よく使う選択肢：")
        cols = st.columns(5)
        for i, opt in enumerate(["あり", "なし", "不明", "未定", "確認中"]):
            if cols[i].button(opt, key=f"q_{opt}", use_container_width=True):
                add_display("user", opt)
                invoke_hearing(opt)
                st.rerun()

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
        if c1.button("🔄 新しい案件を開始する", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
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
def render_policy_review(idata: dict):
    st.info(
        "AIがヒアリング情報を分析し、調査方針を作成しました。以下をご確認ください。\n\n"
        "⚠️ **不明・未定情報がある場合は必ずご確認ください**：情報が不足していると、"
        "必要な法令確認や届出施設確認ができず、見落としが発生するリスクがあります。"
        "「調査前に追記する」から情報を追記してから調査を開始することを推奨します。\n\n"
        "● **内容に問題がない場合**：「この方針で調査を開始する」を押してください。"
    )
    with st.expander("📋 ヒアリング情報のまとめ", expanded=True):
        st.write(idata.get("analysis_summary", ""))

    with st.expander("🔍 調査が必要な項目", expanded=True):
        for issue in idata.get("issues", []):
            st.write(f"● {issue}")

    if idata.get("unknown_items"):
        st.markdown('<p style="color:red;font-weight:bold;font-size:16px;">⚠️ 確認した方がいい不明・未定情報（調査前に追記した方がいい不明・未定情報）</p>', unsafe_allow_html=True)
        for u in idata["unknown_items"]:
            st.markdown(
                f'<div style="border:1.5px solid #e57373;border-radius:6px;padding:10px 14px;margin-bottom:8px;background:#fff8f8;">'
                f'❓ {u}</div>',
                unsafe_allow_html=True,
            )

    with st.expander("📌 調査方針", expanded=True):
        st.write(idata.get("search_plan", ""))
        st.caption("検索キーワード（e-Gov法令API）: " + " / ".join(idata.get("search_keywords", [])))
        st.caption("🌐 AIWeb検索（自動実施） ● 横浜市・神奈川県の条例・規制（公式サイト） ● 省庁ガイドライン・FAQ（厚生労働省・消防庁・環境省・国土交通省・経済産業省）")

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
                    resume_graph(f"approved: {note}")
                    st.rerun()
    with c2:
        if st.button("✅　この方針で調査を開始する", type="primary", use_container_width=True):
            add_display("user", "調査方針を承認しました。調査を開始してください。")
            add_display("system", "AI調査中．．．")
            resume_graph("approved")
            st.rerun()


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

    _pending_fetches = []
    for _law in law_items:
        _lid = _law.get("law_id", "") or _law_id_map.get(_law.get("law_name", ""), "")
        _rid = _law.get("law_revision_id", "") or _law_rev_id_map.get(_law.get("law_name", ""), "")
        _arts = [a for a in _law.get("relevant_articles", []) if a and a.strip()]
        _cache_k = f"article_text_{_lid}"
        if _lid and _arts and _cache_k not in st.session_state:
            _pending_fetches.append((_lid, _rid, _arts, _cache_k))
    if _pending_fetches:
        with st.spinner(f"条文を取得中...（{len(_pending_fetches)}件）"):
            for _lid, _rid, _arts, _ck in _pending_fetches:
                st.session_state[_ck] = fetch_article_text(_lid, _arts, _rid)

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
                f'📌 <em>{law.get("applicability", "")}</em></div>',
                unsafe_allow_html=True,
            )

            # 条番号 + 届出施設 サマリーカード
            cache_key = f"article_text_{law_id}"
            cached_texts: dict = st.session_state.get(cache_key, {}) if law_id else {}

            def _art_link(a: str) -> str:
                if law_id:
                    return (f'<a href="https://laws.e-gov.go.jp/law/{law_id}" target="_blank" '
                            f'style="color:#1565C0;text-decoration:underline;">{a}</a>')
                return f'<span style="color:#1565C0;">{a}</span>'

            art_str = "　".join(_art_link(a) for a in relevant_articles) \
                if relevant_articles else '<span style="color:#888;">条番号確認中</span>'
            auth_str = "　/　".join(authorities) if authorities else '<span style="color:#888;">―</span>'

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

            # 条文インライン表示
            egov_link_url = (
                f"https://laws.e-gov.go.jp/law/{law_id}" if law_id
                else f"https://laws.e-gov.go.jp/search?lawname={urllib.parse.quote(law_name)}"
            )
            egov_link_html = (
                f'<a href="{egov_link_url}" target="_blank" '
                f'style="font-size:12px;color:#1565C0;">🔗 e-Gov で条文を確認</a>'
            )
            if cached_texts:
                art_rows = "".join(
                    f'<div style="margin-bottom:10px;">'
                    f'<div style="font-size:12px;font-weight:700;color:#1565C0;margin-bottom:3px;">{ref}</div>'
                    f'<div style="font-size:12px;color:#333;line-height:1.75;white-space:pre-wrap;">{text}</div>'
                    f'</div>'
                    for ref, text in cached_texts.items()
                )
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
                        f'<span style="color:#283593;">📖 {law_name}&nbsp;{article_ref}</span>'
                        if article_ref else
                        f'<span style="color:#888;">📖 条文番号確認中</span>'
                    )
                    authority_val = d.get("authority", "")
                    st.markdown(
                        f'<div style="background:{d_bg};border-left:4px solid {d_border};'
                        f'padding:9px 13px;border-radius:4px;margin:4px 0;">'
                        f'<div style="font-weight:600;font-size:14px;">{d_icon} {d.get("item", "")}</div>'
                        f'<div style="margin-top:5px;font-size:12px;color:#555;display:flex;flex-wrap:wrap;gap:12px;">'
                        f'{article_html}'
                        f'<span style="color:#1B5E20;font-weight:600;">🏛️ {authority_val}</span>'
                        f'<span>⏰ {d.get("deadline", "")}</span>'
                        f'</div>'
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
                        f'● <strong>{act.get("item", "")}</strong><br>'
                        f'<span style="font-size:12px;color:#555;">⏰ 期限：{act.get("deadline", "")}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


def render_results_review(idata: dict):
    render_results_detail(idata)

    st.divider()
    if st.button("📝　内容を確認してレポートを作成する", type="primary", use_container_width=True):
        add_display("user", "結果レビューが完了しました。レポートを作成してください。")
        resume_graph({"decisions": {}})
        st.rerun()


# ─────────────────────────────────────────
# レポートレビュー UI
# ─────────────────────────────────────────
def render_report_review(idata: dict):
    # ステップ5（結果確認）の詳細を引き続き表示し、消えないようにする
    results_idata = st.session_state.get("results_idata")
    if results_idata:
        st.markdown(
            '<div class="phase-banner">✅　5. 結果確認</div>',
            unsafe_allow_html=True,
        )
        render_results_detail(results_idata)
        st.divider()
        st.markdown(
            '<div class="phase-banner">📄　6. レポート作成</div>',
            unsafe_allow_html=True,
        )

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
        if st.button("✅　確認完了・承認する", type="primary", use_container_width=True):
            st.session_state.report_html = report_html  # complete画面でも参照できるよう保存
            add_display("user", "レポートを確認しました。承認します。")
            resume_graph("approved")
            st.rerun()
    with col2:
        with st.expander("✏️ 修正フィードバック"):
            with st.form("report_feedback"):
                fb = st.text_area("修正内容：", placeholder="例：○○の対応内容の表現を変えてほしい")
                if st.form_submit_button("フィードバックして承認"):
                    add_display("user", f"修正要望あり：{fb}")
                    resume_graph(f"approved_with_feedback: {fb}")
                    st.rerun()


# ─────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚖️ ステップ")

        idata = st.session_state.interrupt_data
        if idata:
            phase_idx = PHASE_INDEX.get(idata.get("phase", ""), 0)
        elif st.session_state.ui_phase == "complete":
            phase_idx = 6
        elif st.session_state.ui_phase == "hearing":
            phase_idx = 0
        else:
            phase_idx = 0

        st.progress(phase_idx / len(PHASES))
        st.caption(f"ステップ {phase_idx + 1} / {len(PHASES)}")

        for i, phase in enumerate(PHASES):
            if i < phase_idx:
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
                    f'<div class="sidebar-card">{badge} {icon} <b>{label}</b><br>{val}</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.caption(f"案件ID: {st.session_state.thread_id[:8]}...")
        st.caption("⚠️ 本ツールの提案は参考情報です。最終判断は担当者が行ってください。")


# ─────────────────────────────────────────
# レポートプレビュー（インライン）
# ─────────────────────────────────────────
def render_report_modal():
    if not st.session_state.get("show_report"):
        return
    html = st.session_state.get("report_html", "")
    if not html:
        return

    with st.expander("📄 Webレポート（プレビュー）", expanded=True):
        st.iframe(html, height=620, scrolling=True)
        if st.button("閉じる"):
            st.session_state.show_report = False
            st.rerun()


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

    render_report_modal()
    st.divider()

    if not st.session_state.display_messages:
        st.markdown("""
<div style="background:#E8F5E9;border-left:4px solid #43A047;padding:14px 18px;border-radius:4px;margin-bottom:16px">
👋 <b>ようこそ！</b> 法令確認・届出施設確認AIです。<br>
設備情報をヒアリングし、横浜市内の会社施設で設備を導入する際に必要な法令・届出施設をAIが調査・報告します。<br>
まず「開始する」を押すと、AIが設備情報のヒアリングを開始します。<br><br>
<b>進め方：</b> 1. ヒアリング → 2. 情報整理 → 3. 調査方針確認 → 4. 調査実施 → 5. 結果確認 → 6. レポート作成 → 7. 完了
</div>
""", unsafe_allow_html=True)
    else:
        render_messages()

    st.divider()
    render_input()


if __name__ == "__main__":
    main()
