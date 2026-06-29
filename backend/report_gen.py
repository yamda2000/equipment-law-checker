"""HTMLレポート生成"""

from datetime import datetime
import json
import uuid
import urllib.parse

from backend.tools.egov import fetch_article_text


PRIORITY_CONFIG = {
    "required": {"label": "🔴 必須対応", "color": "#FFEBEE", "border": "#C62828"},
    "check":    {"label": "🟡 要確認",   "color": "#FFFDE7", "border": "#F57F17"},
}


def generate_html_report(
    equipment_info: dict,
    law_items: list,
    unknown_items: list,
    search_results: list,
    case_id: str = None,
) -> str:
    """法令別アクションアイテムから HTML レポートを生成する"""
    now = datetime.now()
    case_id = case_id or f"EQ-{now.strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"

    info_rows    = _build_info_rows(equipment_info)
    summary_html = _build_summary(law_items)
    law_html     = _build_law_items(law_items)
    unknown_html = _build_unknown_items(unknown_items)
    law_refs     = _build_law_refs(search_results)

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>法令・届出施設確認レポート {case_id}</title>
<style>
  body {{ font-family: 'Noto Sans JP', 'メイリオ', sans-serif; font-size: 14px;
          color: #212121; background: #fafafa; margin: 0; padding: 24px; }}
  .container {{ max-width: 960px; margin: 0 auto; background: white;
                padding: 32px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.1); }}
  h1 {{ color: #1565C0; border-bottom: 3px solid #1565C0; padding-bottom: 8px; font-size: 22px; }}
  h2 {{ color: #1565C0; font-size: 17px; margin-top: 32px; border-left: 4px solid #1565C0; padding-left: 10px; }}
  h3 {{ font-size: 14px; color: #333; margin: 12px 0 6px; }}
  .meta-table, .info-table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
  .meta-table td, .info-table td {{ border: 1px solid #e0e0e0; padding: 8px 12px; }}
  .meta-table td:first-child, .info-table td:first-child {{
    background: #E8EAF6; font-weight: 600; width: 30%; }}
  .law-card {{ border-radius: 6px; padding: 16px; margin: 12px 0; border-left: 5px solid; }}
  .law-title {{ font-weight: 700; font-size: 16px; margin-bottom: 4px; }}
  .law-applicability {{ font-size: 12px; color: #555; margin-bottom: 12px; }}
  .section-label {{ font-weight: 700; font-size: 13px; margin: 10px 0 6px; }}
  .delivery-row {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 4px;
                   padding: 8px 12px; margin: 4px 0; font-size: 13px; }}
  .delivery-meta {{ font-size: 11px; color: #666; margin-top: 2px; }}
  .internal-row {{ background: #F8F9FA; border-left: 3px solid #6B7280;
                   padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 13px; }}
  .internal-meta {{ font-size: 11px; color: #666; margin-top: 2px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
             font-size: 11px; font-weight: 700; color: white; }}
  .badge-required {{ background: #C62828; }}
  .badge-check    {{ background: #F57F17; }}
  .badge-pending  {{ background: #1565C0; }}
  .review-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
                   font-size: 11px; background: #E8F5E9; color: #2E7D32; margin-left: 8px; }}
  .unknown-item {{ background: #FFF8E1; border-left: 4px solid #FFA000;
                   padding: 8px 12px; margin: 6px 0; border-radius: 4px; font-size: 13px; }}
  .law-ref {{ background: #F5F5F5; padding: 8px 12px; margin: 4px 0;
               border-radius: 4px; font-size: 12px; }}
  .law-ref a {{ color: #1565C0; text-decoration: none; }}
  .disclaimer {{ background: #FFF3E0; border: 1px solid #FF9800; padding: 16px;
                  border-radius: 6px; font-size: 13px; margin-top: 32px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0; }}
  .summary-box {{ text-align: center; padding: 16px; border-radius: 8px; }}
  .summary-box .num {{ font-size: 36px; font-weight: 900; }}
  @media print {{ body {{ padding: 0; background: white; }}
                  .container {{ box-shadow: none; padding: 16px; }} }}
</style>
</head>
<body>
<div class="container">

<h1>⚖️ 設備導入時 法令・届出施設確認レポート</h1>

<table class="meta-table">
  <tr><td>案件ID</td><td>{case_id}</td></tr>
  <tr><td>作成日時</td><td>{now.strftime('%Y年%m月%d日 %H:%M')}</td></tr>
  <tr><td>調査対象</td><td>横浜市内会社施設</td></tr>
</table>

<h2>📋 設備情報</h2>
<table class="info-table">{info_rows}</table>

<h2>📊 対応サマリー</h2>
{summary_html}

<h2>⚖️ 法令別 届出・対応事項</h2>
{law_html}

{f'<h2>⚠️ 不明・未定情報（追加確認タスク）</h2>{unknown_html}' if unknown_items else ''}

{f'<h2>📚 参照した法令・情報源</h2>{law_refs}' if law_refs else ''}

<div class="disclaimer">
  <strong>⚠️ 重要な注意事項</strong><br>
  本レポートは、AIが入力情報と検索時点の情報をもとに作成した参考情報です。<br>
  <ul>
    <li>最終的な法令判断・届出要否の確定は、設備導入担当者・関係部署・所轄機関が行ってください。</li>
    <li>地方条例・消防本部の運用基準・行政指導は、本レポートに含まれない場合があります。</li>
    <li>稼働開始前に、法令改正・条例改正の有無を再確認することを推奨します。</li>
    <li>不明・未定情報が確定した場合は、再調査を実施してください。</li>
  </ul>
  <strong>調査制約：</strong>
  e-Gov API（国法令）／公開Web情報（Gemini Google Search）／許可された社内文書 の範囲内で調査しています。
</div>

</div>
</body>
</html>"""


def _build_info_rows(info: dict) -> str:
    labels = {
        "equipment_type":     "設備種別",
        "installation_place": "設置場所",
        "operation_purpose":  "用途・目的",
        "scheduled_date":     "稼働開始予定",
        "chemicals":          "薬品・ガス",
        "fire_exhaust":       "火気・排気",
        "wastewater":         "排水",
        "noise_vibration":    "騒音・振動",
        "radiation":          "放射線・X線",
        "construction":       "建屋改修",
        "additional_info":    "その他情報",
    }
    rows = ""
    for key, label in labels.items():
        val = info.get(key, "―")
        rows += f"<tr><td>{label}</td><td>{val}</td></tr>"
    return rows


def _build_summary(law_items: list) -> str:
    cnt = {"required": 0, "check": 0}
    for law in law_items:
        p = law.get("priority", "check")
        if p in cnt:
            cnt[p] += 1

    def box(priority, label, color, text_color):
        n = cnt[priority]
        return (f'<div class="summary-box" style="background:{color}">'
                f'<div class="num" style="color:{text_color}">{n}</div>'
                f'<div style="font-size:13px;color:{text_color}">{label}</div></div>')

    return (f'<div class="summary-grid" style="grid-template-columns:repeat(2,1fr);">'
            f'{box("required", "🔴 必須対応", "#FFEBEE", "#C62828")}'
            f'{box("check",    "🟡 要確認",   "#FFFDE7", "#F57F17")}'
            f'</div>')


def _build_article_block(law: dict) -> str:
    """条番号＋届出施設のサマリーカードと条文インライン表示を生成する。
    （ステップ5「結果確認」の表示とフォーマットを統一）"""
    law_name = law.get("law_name", "")
    law_id = law.get("law_id", "")
    law_revision_id = law.get("law_revision_id", "")
    relevant_articles = [a for a in law.get("relevant_articles", []) if a and a.strip()]

    # 届出施設を重複なしで収集
    authorities = list(dict.fromkeys(
        d.get("authority", "") for d in law.get("deliveries", []) if d.get("authority", "")
    ))

    # 条文テキストを取得（e-Gov）。XMLは結果確認時にキャッシュ済みのため通常は即時。
    article_texts: dict = {}
    if law_id and relevant_articles:
        try:
            article_texts = fetch_article_text(law_id, relevant_articles, law_revision_id)
        except Exception:
            article_texts = {}
    # error キーは表示対象外
    article_texts = {k: v for k, v in (article_texts or {}).items() if k != "error"}

    def _art_link(a: str) -> str:
        if law_id:
            return (f'<a href="https://laws.e-gov.go.jp/law/{law_id}" target="_blank" '
                    f'style="color:#1565C0;text-decoration:underline;">{a}</a>')
        return f'<span style="color:#1565C0;">{a}</span>'

    art_str = "　".join(_art_link(a) for a in relevant_articles) \
        if relevant_articles else '<span style="color:#888;">条番号確認中</span>'
    auth_str = "　/　".join(authorities) if authorities else '<span style="color:#888;">―</span>'

    summary_card = (
        f'<div style="border:1px solid #E0E0E0;border-radius:6px;overflow:hidden;margin:10px 0;">'
        f'<div style="display:flex;align-items:flex-start;background:#EFF3FF;border-bottom:1px solid #E0E0E0;">'
        f'<div style="padding:7px 12px;color:#1565C0;font-weight:700;font-size:13px;min-width:90px;white-space:nowrap;">📖 条番号</div>'
        f'<div style="padding:7px 12px;font-size:13px;">{art_str}</div>'
        f'</div>'
        f'<div style="display:flex;align-items:center;background:#F3FBF0;">'
        f'<div style="padding:7px 12px;color:#2E7D32;font-weight:700;font-size:13px;min-width:90px;white-space:nowrap;">🏛️ 届出施設</div>'
        f'<div style="padding:7px 12px;font-size:13px;color:#1B5E20;font-weight:600;">{auth_str}</div>'
        f'</div>'
        f'</div>'
    )

    if article_texts:
        art_rows = "".join(
            f'<div style="margin-bottom:10px;">'
            f'<div style="font-size:12px;font-weight:700;color:#1565C0;margin-bottom:3px;">{ref}</div>'
            f'<div style="font-size:12px;color:#333;line-height:1.75;white-space:pre-wrap;">{text}</div>'
            f'</div>'
            for ref, text in article_texts.items()
        )
        article_box = (
            f'<div style="background:#F8F9FA;border-left:3px solid #1565C0;'
            f'border-radius:0 4px 4px 0;padding:10px 14px;margin:10px 0;">'
            f'<div style="font-size:11px;color:#888;margin-bottom:6px;">📜 条文（e-Gov）</div>'
            f'{art_rows}'
            f'</div>'
        )
    else:
        egov_url = (
            f"https://laws.e-gov.go.jp/law/{law_id}" if law_id
            else f"https://laws.e-gov.go.jp/search?lawname={urllib.parse.quote(law_name)}"
        )
        article_box = (
            f'<div style="margin:8px 0;"><a href="{egov_url}" target="_blank" '
            f'style="font-size:12px;color:#1565C0;">🔗 e-Gov で条文を確認</a></div>'
        )

    return summary_card + article_box


def _build_law_items(law_items: list) -> str:
    if not law_items:
        return "<p>対応事項はありません。</p>"

    html = ""
    for law in law_items:
        p = law.get("priority", "check")
        cfg = PRIORITY_CONFIG.get(p, PRIORITY_CONFIG["check"])
        badge_class = f"badge-{p}"
        review = law.get("review_decision", "")
        review_html = f'<span class="review-badge">✅ {review}</span>' if review else ""

        article_block = _build_article_block(law)

        relevant_articles = law.get("relevant_articles", [])
        deliveries_html = ""
        for d in law.get("deliveries", []):
            dp = d.get("priority", "check")
            dcfg = PRIORITY_CONFIG.get(dp, PRIORITY_CONFIG["check"])
            article_ref = d.get("law_article", "") or "・".join(relevant_articles)
            article_html = (
                f'<div style="font-size:11px;color:#283593;margin:2px 0;">📖 根拠条文：<strong>{law.get("law_name","")} {article_ref}</strong></div>'
                if article_ref else
                f'<div style="font-size:11px;color:#888;margin:2px 0;">📖 根拠条文：条文番号不明（e-Gov で要確認）</div>'
            )
            deliveries_html += (
                f'<div class="delivery-row">'
                f'<span class="badge badge-{dp}">{dcfg["label"]}</span> {d.get("item", "")}'
                f'{article_html}'
                f'<div class="delivery-meta">🏛️ 届出先：{d.get("authority", "")}　⏰ 期限：{d.get("deadline", "")}</div>'
                f'</div>'
            )

        internal_html = ""
        for act in law.get("internal_actions", []):
            internal_html += (
                f'<div class="internal-row">'
                f'● {act.get("item", "")}'
                f'<div class="internal-meta">⏰ 期限：{act.get("deadline", "")}</div>'
                f'</div>'
            )

        html += f"""
<div class="law-card" style="background:{cfg['color']};border-color:{cfg['border']}">
  <div class="law-title">
    <span class="badge {badge_class}">{cfg['label']}</span>
    {law.get('law_name', '')}
    {review_html}
  </div>
  <div class="law-applicability">📌 {law.get('applicability', '')}</div>
  {article_block}
  {'<div class="section-label">📋 届出・申請事項</div>' + deliveries_html if deliveries_html else '<div class="section-label" style="color:#888">📋 届出・申請事項なし（要確認）</div>'}
  {'<div class="section-label">🏢 社内対応事項</div>' + internal_html if internal_html else ''}
</div>"""
    return html


def _build_unknown_items(items: list) -> str:
    if not items:
        return ""
    return "".join(
        f'<div class="unknown-item">❓ {item}</div>' for item in items
    )


def _build_law_refs(search_results: list) -> str:
    refs = [r for r in search_results if r.get("title") and "error" not in r.get("source", "")]
    if not refs:
        return ""
    html = ""
    for r in refs[:15]:
        url = r.get("url", "")
        title = r.get("title", "")
        source = r.get("source", "")
        link = f'<a href="{url}" target="_blank">{title}</a>' if url else title
        html += f'<div class="law-ref">📖 {link} <span style="color:#888">({source})</span></div>'
    return html
