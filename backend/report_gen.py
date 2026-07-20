"""HTMLレポート生成"""

from datetime import datetime
import html as html_lib
import json
import re
import uuid
import urllib.parse

from backend.tools.egov import (
    fetch_article_text, fetch_article_captions, fetch_article_chapters,
    article_sort_key, get_suggested_keywords, resolve_law_alias,
)


PRIORITY_CONFIG = {
    "required": {"label": "🔴 必須対応", "color": "#FFEBEE", "border": "#C62828"},
    "check":    {"label": "🟡 要確認",   "color": "#FFFDE7", "border": "#F57F17"},
}

# LLM出力に英字の内部フィールド名が混ざった場合の表示用置換
_FIELD_NAME_JA = {
    "equipment_type":     "設備種別",
    "installation_place": "設置場所",
    "operation_purpose":  "用途・目的",
    "scheduled_date":     "稼働開始予定",
    "chemicals":          "薬品・ガス",
    "fire_exhaust":       "火気・排気・粉じん",
    "wastewater":         "排水・廃棄物",
    "noise_vibration":    "騒音・振動",
    "radiation":          "放射線・X線",
    "construction":       "建屋改修",
    "additional_info":    "その他情報",
}


def _to_ja_fields(text: str) -> str:
    for en, ja in _FIELD_NAME_JA.items():
        text = str(text).replace(en, f"「{ja}」")
    return text


def _esc(text) -> str:
    """LLM出力・e-Gov条文・Web検索スニペット等をHTMLに挿入する前のエスケープ。"""
    return html_lib.escape(str(text or ""))


# 条例（e-Gov 未収載）の原文を確認できる公式例規集
ORDINANCE_DB_LINKS = [
    ("横浜市",   "横浜市例規集",   "https://cgi.city.yokohama.lg.jp/somu/reiki/reiki_menu.html"),
    ("神奈川県", "神奈川県法規集", "https://www.pref.kanagawa.jp/docs/y8e/cnt/f7406/"),
]

# 頻出条例の本文直リンク。例規集システムの更新でURLが変わる可能性があるため、
# 例規集トップへのリンクと併用する
ORDINANCE_DIRECT_URLS = {
    "横浜市生活環境の保全等に関する条例":
        "https://cgi.city.yokohama.lg.jp/somu/reiki/reiki_honbun/g202RG00001294.html",
    "横浜市火災予防条例":
        "https://cgi.city.yokohama.lg.jp/somu/reiki/reiki_honbun/g202RG00001081.html",
}


def ordinance_links_html(law_name: str) -> str:
    """e-Gov 未収載の条例向けに、公式例規集へのリンクHTMLを返す。
    法令名から自治体を判別できない場合は横浜市・神奈川県の両方を出す。"""
    links = []
    direct = ORDINANCE_DIRECT_URLS.get(law_name)
    if direct:
        links.append(
            f'<a href="{direct}" target="_blank" style="color:#1565C0;">'
            f'🔗 例規集で条文を確認</a>'
        )
    dbs = [(label, url) for key, label, url in ORDINANCE_DB_LINKS if key in law_name]
    if not dbs:
        dbs = [(label, url) for _, label, url in ORDINANCE_DB_LINKS]
    links += [
        f'<a href="{url}" target="_blank" style="color:#1565C0;">📚 {label}</a>'
        for label, url in dbs
    ]
    return "　".join(links)


def generate_html_report(
    equipment_info: dict,
    law_items: list,
    unknown_items: list,
    search_results: list,
    case_id: str = None,
    feedback_note: str = "",
    excluded_laws: list = None,
    uncovered_issues: list = None,
    summary: str = "",
    issues: list = None,
    issue_coverage: dict = None,
    coverage_check_failed: bool = False,
    web_search_unconfirmed: bool = False,
) -> str:
    """法令別アクションアイテムから HTML レポートを生成する"""
    now = datetime.now()
    case_id = case_id or f"EQ-{now.strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

    feedback_html = (
        f'<div style="background:#E3F2FD;border-left:4px solid #1565C0;padding:14px 18px;'
        f'border-radius:6px;margin:16px 0;font-size:13px;color:#0D47A1;">'
        f'<strong>📝 担当者からの補足・修正メモ</strong><br>{_esc(feedback_note)}</div>'
        if feedback_note else ""
    )

    info_rows      = _build_info_rows(equipment_info)
    intro_html     = _build_intro(summary)
    checklist_html = _build_action_checklist(
        law_items, (equipment_info or {}).get("scheduled_date", ""))
    summary_html   = _build_summary(law_items)
    law_html       = _build_law_items(law_items)
    unknown_html   = _build_unknown_items(unknown_items)
    law_refs       = _build_law_refs(search_results)
    uncovered_html = _build_coverage_check(
        issues or [], uncovered_issues or [], issue_coverage or {},
        check_failed=coverage_check_failed,
    )
    web_unconfirmed_html = _build_web_unconfirmed_notice(web_search_unconfirmed)
    matrix_html    = _build_law_matrix(law_items, excluded_laws or [], equipment_info)

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
  .intro-box {{ background: #EFF3FF; border: 1px solid #C5CAE9; border-left: 5px solid #1565C0;
                border-radius: 6px; padding: 14px 18px; font-size: 13px; margin: 12px 0; }}
  .intro-box ul {{ margin: 8px 0 0; padding-left: 20px; }}
  .intro-box li {{ margin: 3px 0; }}
  .checklist {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 13px; }}
  .checklist th {{ background: #E8EAF6; border: 1px solid #e0e0e0; padding: 6px 8px;
                   font-size: 12px; text-align: left; }}
  .checklist td {{ border: 1px solid #e0e0e0; padding: 6px 8px; vertical-align: top; }}
  .checklist .cat td {{ background: #1565C0; color: white; font-weight: 700; font-size: 13px; }}
  .checklist .cb {{ text-align: center; width: 30px; font-size: 16px; color: #555; }}
  .checklist .cb input {{ width: 16px; height: 16px; cursor: pointer; margin: 2px 0 0; }}
  .checklist tr.done td {{ opacity: .55; }}
  .checklist tr.done td:nth-child(2) {{ text-decoration: line-through; }}
  .csv-btn {{ background: #1565C0; color: white; border: none; border-radius: 6px;
              padding: 8px 14px; font-size: 13px; font-weight: 700; cursor: pointer; }}
  .csv-btn:hover {{ background: #1976D2; }}
  @media print {{ body {{ padding: 0; background: white; }}
                  .container {{ box-shadow: none; padding: 16px; }}
                  .csv-btn {{ display: none; }} }}
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

{intro_html}

{feedback_html}

<h2>📋 設備情報</h2>
<table class="info-table">{info_rows}</table>

<h2>📊 対応サマリー</h2>
{summary_html}

{checklist_html}

{uncovered_html}

{web_unconfirmed_html}

{matrix_html}

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
        "fire_exhaust":       "火気・排気・粉じん",
        "wastewater":         "排水・廃棄物",
        "noise_vibration":    "騒音・振動",
        "radiation":          "放射線・X線",
        "construction":       "建屋改修",
        "additional_info":    "その他情報",
    }
    rows = ""
    for key, label in labels.items():
        val = info.get(key, "―")
        rows += f"<tr><td>{label}</td><td>{_esc(val)}</td></tr>"
    return rows


def _build_intro(summary: str) -> str:
    """レポート冒頭の「はじめにお読みください」ブロック。
    法令に詳しくない担当者向けに、総括とバッジの読み方を最初に示す。"""
    summary_p = (
        f'<p style="margin:0 0 10px;line-height:1.8;"><strong>【総括】</strong>'
        f'{_esc(_to_ja_fields(summary))}</p>'
        if summary else ""
    )
    return (
        f'<h2>📖 はじめにお読みください（このレポートの読み方）</h2>'
        f'<div class="intro-box">'
        f'{summary_p}'
        f'<ul>'
        f'<li><span class="badge badge-required">🔴 必須対応</span>　この設備に適用され、'
        f'<strong>稼働前に届出・対応が必要</strong>と判断した法令です。</li>'
        f'<li><span class="badge badge-check">🟡 要確認</span>　適用される可能性があり、'
        f'<strong>仕様・数量などの確定後に判定が必要</strong>な法令です。放置せず、記載の確認事項を確かめてください。</li>'
        f'<li>まず「✅ 対応チェックリスト」で<strong>何を・いつまでに・どこへ</strong>を確認し、'
        f'詳細と根拠条文は「⚖️ 法令別 届出・対応事項」をご覧ください。</li>'
        f'<li>本レポートはAIによる参考情報です。最終判断は担当部署・所轄機関にご確認ください。</li>'
        f'</ul>'
        f'</div>'
    )


# 対応チェックリストの期限カテゴリ（表示順）
_TIMELINE_ORDER = ["設置・工事の前", "稼働開始の前", "稼働後・定期", "期限未確定（要確認）"]


def _parse_ym(text: str):
    """「2025年4月1日」等から (年, 月) を取り出す。見つからなければ None。"""
    m = re.search(r"(\d{4})年\s*(\d{1,2})月", str(text or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _deadline_category(deadline: str, sched_ym: tuple = None) -> str:
    """期限の文言を時系列カテゴリに分類する。

    判定順：
    1. 工事・設置・発注など「設置前」を示す語（括弧内の補足を含む）
    2. 稼働・使用開始などの「前」
    3. 「稼働後」「設置後」等の主語つきの「後」・定期表現
       （「機種確定後すみやかに」のような単独の「後」には反応させない）
    4. 具体的な日付 → 稼働開始予定（sched_ym）と年月で比較して前後を判定
    どれにも当たらないものだけを「期限未確定」とする。
    """
    d = str(deadline or "")
    if ("工事" in d and "前" in d) or ("設置" in d and "前" in d) or "着工" in d \
            or "発注前" in d or "施工計画" in d or "調達前" in d or "契約前" in d:
        return "設置・工事の前"
    if (("稼働" in d or "使用開始" in d or "運転開始" in d or "消費開始" in d) and "前" in d) \
            or re.search(r"(ヶ月|か月|カ月|日)前", d):
        return "稼働開始の前"
    if re.search(r"(稼働|設置|使用開始|運転開始|導入|工事完了|竣工)後", d) \
            or "毎年" in d or "定期" in d or "年度" in d or "以内" in d:
        return "稼働後・定期"
    ym = _parse_ym(d)
    if ym:
        if sched_ym:
            return "稼働開始の前" if ym <= sched_ym else "稼働後・定期"
        # 稼働開始予定が不明でも、期日が書けている以上「期限未確定」ではない。
        # この種の期限はほぼ導入準備タスクのため「稼働開始の前」に寄せる。
        return "稼働開始の前"
    return "期限未確定（要確認）"


# チェックリストの完了状態保存（localStorage・端末ごと）と CSV エクスポート。
# レポートは単体のHTMLファイルとして配布されるため、外部ライブラリに依存しない。
_CHECKLIST_SCRIPT = """<script>
(function () {
  var table = document.getElementById('action-checklist');
  if (!table) return;
  var storeKey = 'lawcheck-checklist:' + document.title;
  var state = {};
  try { state = JSON.parse(localStorage.getItem(storeKey) || '{}'); } catch (e) {}

  var category = '';
  var items = [];
  table.querySelectorAll('tr').forEach(function (tr) {
    if (tr.classList.contains('cat')) {
      category = tr.textContent.replace(/^▼\\s*/, '').replace(/（\\d+件）$/, '').trim();
      return;
    }
    var box = tr.querySelector('input.cl-check');
    if (!box) return;
    var cells = tr.querySelectorAll('td');
    // 行番号ではなく「時期＋対応事項」をキーにする（再生成後も同じ項目なら状態を引き継ぐ）
    var key = category + '|' + cells[1].textContent.trim();
    var apply = function () { tr.classList.toggle('done', box.checked); };
    box.checked = !!state[key];
    apply();
    box.addEventListener('change', function () {
      state[key] = box.checked;
      apply();
      try { localStorage.setItem(storeKey, JSON.stringify(state)); } catch (e) {}
    });
    items.push({ cells: cells, box: box, category: category });
  });

  var btn = document.getElementById('cl-csv-btn');
  if (btn) btn.addEventListener('click', function () {
    var esc = function (v) {
      var s = String(v || '');
      // Excel数式インジェクション対策: = + - @ やタブ等で始まるセルは
      // 先頭に ' を付けて数式として評価されないようにする
      if (/^[=+\-@\t\r]/.test(s.trim())) { s = "'" + s; }
      return '"' + s.replace(/"/g, '""') + '"';
    };
    var lines = [['状態', '時期', '対応事項', '種別', '届出先・担当', '期限', '根拠法令'].map(esc).join(',')];
    items.forEach(function (it) {
      var c = it.cells;
      lines.push([
        it.box.checked ? '済' : '未',
        it.category,
        c[1].textContent.trim(),
        c[2].textContent.trim(),
        c[3].textContent.trim(),
        c[4].textContent.trim(),
        c[5].textContent.trim(),
      ].map(esc).join(','));
    });
    // BOM付きUTF-8（Excelでの文字化け防止）
    var blob = new Blob(['\\ufeff' + lines.join('\\r\\n')], { type: 'text/csv;charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (document.title || '法令確認レポート') + '_チェックリスト.csv';
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  });
})();
</script>"""


def _build_action_checklist(law_items: list, scheduled_date: str = "") -> str:
    """全法令の届出・社内対応を期限の時系列で1つの表に統合する。
    「結局、何を・いつまでに・どこへ」に一目で答えるための一覧。
    scheduled_date（稼働開始予定）は日付だけの期限を前後に振り分ける基準に使う。"""
    sched_ym = _parse_ym(scheduled_date)
    rows_by_cat: dict = {c: [] for c in _TIMELINE_ORDER}
    for law in law_items:
        law_name = law.get("law_name", "")
        for d in law.get("deliveries", []):
            if not d.get("item"):
                continue
            rows_by_cat[_deadline_category(d.get("deadline", ""), sched_ym)].append({
                "kind": "🏛️ 届出",
                "item": d.get("item", ""),
                "to": d.get("authority", ""),
                "deadline": d.get("deadline", ""),
                "law": law_name,
                "priority": d.get("priority", law.get("priority", "check")),
            })
        for a in law.get("internal_actions", []):
            if not a.get("item"):
                continue
            rows_by_cat[_deadline_category(a.get("deadline", ""), sched_ym)].append({
                "kind": "🏢 社内",
                "item": a.get("item", ""),
                "to": a.get("responsible", ""),
                "deadline": a.get("deadline", ""),
                "law": law_name,
                "priority": "internal",
            })

    total = sum(len(v) for v in rows_by_cat.values())
    if total == 0:
        return ""

    _prio_rank = {"required": 0, "check": 1, "pending": 2, "internal": 3}
    body = ""
    for cat in _TIMELINE_ORDER:
        rows = rows_by_cat[cat]
        if not rows:
            continue
        rows.sort(key=lambda r: _prio_rank.get(r["priority"], 1))
        body += f'<tr class="cat"><td colspan="6">▼ {cat}（{len(rows)}件）</td></tr>'
        for r in rows:
            if r["priority"] == "required":
                prio_html = '<span class="badge badge-required">必須</span> '
            elif r["priority"] == "check":
                prio_html = '<span class="badge badge-check">要確認</span> '
            elif r["priority"] == "pending":
                prio_html = '<span class="badge badge-pending">確認中</span> '
            else:
                prio_html = ""
            body += (
                f'<tr>'
                f'<td class="cb"><input type="checkbox" class="cl-check" aria-label="この項目を完了にする"></td>'
                f'<td>{prio_html}{_esc(r["item"])}</td>'
                f'<td style="white-space:nowrap;">{r["kind"]}</td>'
                f'<td>{_esc(r["to"])}</td>'
                f'<td>{_esc(r["deadline"])}</td>'
                f'<td style="font-size:12px;">{_esc(r["law"])}</td>'
                f'</tr>'
            )

    return (
        f'<h2>✅ 対応チェックリスト（時系列）</h2>'
        f'<div style="font-size:12px;color:#666;margin-bottom:8px;">'
        f'すべての法令の届出・社内対応を期限順にまとめた一覧です（全{total}件）。'
        f'チェック欄はブラウザ上でそのまま使え、状態はこの端末のブラウザに自動保存されます'
        f'（別の端末・ブラウザには引き継がれません）。Excel等で管理する場合はCSV保存を、'
        f'紙で使う場合は印刷をご利用ください。詳細・根拠条文は下の「法令別 届出・対応事項」にあります。</div>'
        f'<div style="margin-bottom:8px;">'
        f'<button id="cl-csv-btn" type="button" class="csv-btn">⬇️ チェックリストをCSVで保存（Excel対応）</button>'
        f'</div>'
        f'<table class="checklist" id="action-checklist">'
        f'<tr><th></th><th>対応事項</th><th>種別</th><th>届出先・担当</th><th>期限</th><th>根拠法令</th></tr>'
        f'{body}'
        f'</table>'
        f'{_CHECKLIST_SCRIPT}'
    )


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

    # 条見出し（例：（建築確認））をe-Govから取得（XMLはキャッシュ済みのため通常は即時）
    captions: dict = {}
    if law_id and relevant_articles:
        try:
            captions = fetch_article_captions(law_id, relevant_articles, law_revision_id)
        except Exception:
            captions = {}

    # 条番号→章タイトルのマップ（章立てのない法令は空 dict → 従来通りの平坦表示）
    chapters: dict = {}
    if law_id and relevant_articles:
        try:
            chapters = fetch_article_chapters(law_id, law_revision_id)
        except Exception:
            chapters = {}

    def _with_caption(a: str) -> str:
        # 見出しは e-Gov 取得分（本物）のみ表示する。
        # 旧データに残る LLM 由来の（）付き見出しは信頼できないため取り除く
        base = re.sub(r"（[^）]*）", "", a).strip()
        return base + captions.get(base, captions.get(a, ""))

    def _art_link(a: str) -> str:
        a = _esc(_with_caption(a))
        if law_id:
            return (f'<a href="https://laws.e-gov.go.jp/law/{law_id}" target="_blank" '
                    f'style="color:#1565C0;text-decoration:underline;">{a}</a>')
        return f'<span style="color:#1565C0;">{a}</span>'

    def _chapter_of(a: str) -> str:
        # "第28条の2第1項（見出し）" → "第28条の2" に正規化して章を引く
        m = re.match(r"第\d+条(?:の\d+)*", re.sub(r"（[^）]*）", "", a).strip())
        return chapters.get(m.group(0), "") if m else ""

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
                '再調査で特定できます。e-Gov で直接確認も可能です）'
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
            + (f'<div style="font-size:12px;color:#555;font-weight:700;">【{_esc(ch)}】</div>' if ch else "")
            + '<div style="padding-left:1em;">' + "　".join(_art_link(a) for a in arts) + "</div>"
            + "</div>"
            for ch, arts in _groups.items()
        )
    else:
        art_str = "　".join(_art_link(a) for a in relevant_articles)
    auth_str = "　/　".join(_esc(a) for a in authorities) if authorities else '<span style="color:#888;">―</span>'

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

    # 現在の relevant_articles に含まれる条番号だけを、条番号の昇順で表示する
    art_pairs = [
        (a, article_texts[a])
        for a in sorted(relevant_articles, key=article_sort_key)
        if a in article_texts
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
                    f'margin:6px 0 4px;">【{_esc(_ch)}】</div>'
                )
            _last_ch = _ch
            _rows.append(
                f'<div style="margin-bottom:10px;">'
                f'<div style="font-size:12px;font-weight:700;color:#1565C0;margin-bottom:3px;">{_esc(_with_caption(ref))}</div>'
                f'<div style="font-size:12px;color:#333;line-height:1.75;white-space:pre-wrap;">{_esc(text)}</div>'
                f'</div>'
            )
        art_rows = "".join(_rows)
        # 条文全文は既定で折りたたみ、レポートを流し読みしやすくする
        article_box = (
            f'<details style="background:#F8F9FA;border-left:3px solid #1565C0;'
            f'border-radius:0 4px 4px 0;padding:10px 14px;margin:10px 0;">'
            f'<summary style="cursor:pointer;font-size:12px;color:#1565C0;font-weight:700;">'
            f'📜 条文を表示する（e-Gov原文・{len(art_pairs)}条）</summary>'
            f'<div style="margin-top:8px;">{art_rows}</div>'
            f'</details>'
        )
    else:
        if law_id:
            article_box = (
                f'<div style="margin:8px 0;"><a href="https://laws.e-gov.go.jp/law/{law_id}" '
                f'target="_blank" style="font-size:12px;color:#1565C0;">🔗 e-Gov で条文を確認</a></div>'
            )
        elif is_ordinance:
            article_box = (
                f'<div style="margin:8px 0;font-size:12px;">'
                f'{ordinance_links_html(law_name)}'
                f'<span style="color:#888;">　（e-Gov 未収載のため公式例規集で原文をご確認ください）</span>'
                f'</div>'
            )
        else:
            article_box = (
                f'<div style="margin:8px 0;"><a '
                f'href="https://laws.e-gov.go.jp/search?lawname={urllib.parse.quote(law_name)}" '
                f'target="_blank" style="font-size:12px;color:#1565C0;">🔗 e-Gov で法令名を検索</a></div>'
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
        review_html = f'<span class="review-badge">✅ {_esc(review)}</span>' if review else ""

        article_block = _build_article_block(law)

        relevant_articles = law.get("relevant_articles", [])
        deliveries_html = ""
        for d in law.get("deliveries", []):
            dp = d.get("priority", "check")
            dcfg = PRIORITY_CONFIG.get(dp, PRIORITY_CONFIG["check"])
            article_ref = d.get("law_article", "") or "・".join(relevant_articles)
            article_html = (
                f'<div style="font-size:11px;color:#283593;margin:2px 0;">📖 根拠条文：<strong>{_esc(law.get("law_name",""))} {_esc(article_ref)}</strong></div>'
                if article_ref else
                f'<div style="font-size:11px;color:#888;margin:2px 0;">📖 根拠条文：条文番号不明（e-Gov で要確認）</div>'
            )
            basis = d.get("authority_basis", "")
            src_url = d.get("authority_source_url", "")
            src_title = d.get("authority_source_title") or "出典ページ"
            # Google検索グラウンディング由来のリンク・タイトルは、Gemini API
            # 追加利用規約により保存が制限されるため、恒久保存されるレポートには
            # 掲載しない（結果確認画面では表示される）
            if d.get("authority_source_kind", "") == "Gemini Grounding":
                src_url = ""
                src_title = ""
            src_link = (
                f'　<a href="{_esc(src_url)}" target="_blank" style="color:#1565C0;">🔗 '
                f'{_esc(src_title[:40])}</a>'
                if src_url else ""
            )
            basis_html = (
                f'<div style="font-size:11px;color:#607D8B;margin-top:2px;line-height:1.6;">'
                f'└ 届出先の根拠：{_esc(basis)}{src_link}</div>'
                if (basis or src_url) else ""
            )
            deliveries_html += (
                f'<div class="delivery-row">'
                f'<span class="badge badge-{dp}">{dcfg["label"]}</span> {_esc(d.get("item", ""))}'
                f'{article_html}'
                f'<div class="delivery-meta">🏛️ 届出先：{_esc(d.get("authority", ""))}　⏰ 期限：{_esc(d.get("deadline", ""))}</div>'
                f'{basis_html}'
                f'</div>'
            )

        internal_html = ""
        for act in law.get("internal_actions", []):
            internal_html += (
                f'<div class="internal-row">'
                f'● {_esc(act.get("item", ""))}'
                f'<div class="internal-meta">⏰ 期限：{_esc(act.get("deadline", ""))}</div>'
                f'</div>'
            )

        html += f"""
<div class="law-card" style="background:{cfg['color']};border-color:{cfg['border']}">
  <div class="law-title">
    <span class="badge {badge_class}">{cfg['label']}</span>
    {_esc(law.get('law_name', ''))}
    {review_html}
  </div>
  <div class="law-applicability">📌 {_esc(law.get('applicability', ''))}</div>
  {article_block}
  {'<div class="section-label">📋 届出・申請事項</div>' + deliveries_html if deliveries_html else '<div class="section-label" style="color:#888">📋 届出・申請事項：未特定（「届出不要」の意味ではありません。適用が確定した場合は再調査で特定できます）</div>'}
  {'<div class="section-label">🏢 社内対応事項</div>' + internal_html if internal_html else ''}
</div>"""
    return html


def _build_web_unconfirmed_notice(unconfirmed: bool) -> str:
    """Web検索が1件も実行・成功できなかった場合の明示。
    「Web 0件」を条例・届出先の情報が実際に無かった結果と混同させないための警告。"""
    if not unconfirmed:
        return ""
    return (
        '<div style="background:#FFEBEE;border:1px solid #EF9A9A;border-left:5px solid #C62828;'
        'border-radius:6px;padding:14px 18px;font-size:13px;color:#B71C1C;margin:12px 0;">'
        '<strong>⚠️ Web情報（条例・届出先など）は未確認です：</strong>'
        'この調査ではWeb検索が1件も実行・成功できませんでした（GEMINI_API_KEY未設定、'
        'または継続的なエラー）。e-Gov法令API・社内文書に収載のない条例・届出先の情報は'
        '本レポートに反映されていない可能性があります。担当部署・所轄機関への直接確認を推奨します。'
        '</div>'
    )


def _build_coverage_check(
    issues: list, uncovered: list, coverage: dict = None,
    check_failed: bool = False,
) -> str:
    """網羅性チェックの結果。論点ごとの✅/⚠️判定を明示し、
    OKの場合も「チェックしてOKだった」ことが分かるように表示する。
    coverage は {論点: [カバー元の法令・情報タイトル, ...]}（カバー判定の検証用）。
    check_failed=True はチェック自体が実行できなかった状態。OK表示は絶対に出さない。"""
    if not issues and not uncovered and not check_failed:
        return ""
    coverage = coverage or {}

    covered_n = len([i for i in issues if i not in uncovered])

    if check_failed:
        # チェック未実行：カバー済み/未カバーの区別自体が信頼できないため、
        # 論点別の✅/⚠️テーブルは出さず、未カバー扱いの論点リストのみ表示する
        ng_rows = "".join(
            f'<li style="margin-bottom:6px;line-height:1.6;">{_to_ja_fields(_esc(item))}</li>'
            for item in uncovered
        )
        uncovered_html = (
            f'<br><strong>下記の論点は補完前の時点で未カバーでした（要確認）：</strong>'
            f'<ul style="margin:10px 0 0;padding-left:20px;">{ng_rows}</ul>'
            if uncovered else ""
        )
        summary_box = (
            f'<div style="background:#FFF8E1;border:1px solid #FFE082;border-left:5px solid #F57F17;'
            f'border-radius:6px;padding:14px 18px;font-size:13px;color:#5D4037;">'
            f'<strong>⚠️ チェック未完了：</strong>網羅性チェックを完了できませんでした'
            f'（AI呼び出しエラー）。論点のカバー状況は自動確認されていません。'
            f'<strong>調査論点 {len(issues)}件のカバー状況を手動で確認してください。</strong>'
            f'{uncovered_html}</div>'
        )
        return (
            f'<h2 style="color:#F57F17;border-left-color:#F57F17;">🧮 網羅性チェック（調査論点のカバー状況）</h2>'
            f'<div style="font-size:12px;color:#666;margin-bottom:8px;">'
            f'調査開始前に洗い出した論点ごとに、対応する法令・情報を収集できたかを'
            f'AIの調査完了判断とは独立に検証した結果です。</div>'
            f'{summary_box}'
        )

    # 判定サマリー（OK: 緑 / NG: 赤）
    if issues and not uncovered:
        summary_box = (
            f'<div style="background:#E8F5E9;border:1px solid #A5D6A7;border-left:5px solid #2E7D32;'
            f'border-radius:6px;padding:12px 18px;font-size:13px;color:#1B5E20;">'
            f'<strong>✅ チェック結果 OK：</strong>調査論点 {len(issues)}件すべてについて、'
            f'対応する法令・情報の収集を確認しました。</div>'
        )
    else:
        ng_rows = "".join(
            f'<li style="margin-bottom:6px;line-height:1.6;">{_to_ja_fields(_esc(item))}</li>'
            for item in uncovered
        )
        head = (
            f'調査論点 {len(issues)}件中 {covered_n}件はカバー済み。'
            f'<strong>下記 {len(uncovered)}件は対応する法令・情報を確認できませんでした。</strong>'
            if issues else
            '<strong>以下の論点は、AI調査で対応する法令・情報を確認できませんでした。</strong>'
        )
        summary_box = (
            f'<div style="background:#FFEBEE;border:1px solid #EF9A9A;border-left:5px solid #C62828;'
            f'border-radius:6px;padding:14px 18px;font-size:13px;">'
            f'<strong>🚨 チェック結果 NG：</strong>{head}<br>'
            f'確認漏れ・届出漏れを防ぐため、<strong>担当部署・所轄機関への直接確認</strong>を行ってください。'
            f'<ul style="margin:10px 0 0;padding-left:20px;">{ng_rows}</ul>'
            f'</div>'
        )

    # 論点別の判定一覧
    detail_rows = ""
    for i in issues:
        if i in uncovered:
            detail_rows += (
                f'<tr style="background:#FFEBEE;">'
                f'<td style="white-space:nowrap;color:#B71C1C;font-weight:700;">⚠️ 未カバー</td>'
                f'<td>{_to_ja_fields(_esc(i))}　<span style="color:#B71C1C;">（手動確認を推奨）</span></td></tr>'
            )
        else:
            covered_by = coverage.get(i) or []
            covered_by_html = (
                f'<div style="font-size:11px;color:#2E7D32;margin-top:3px;">'
                f'└ カバー元：{_esc("、".join(covered_by))}</div>'
                if covered_by else
                '<div style="font-size:11px;color:#888;margin-top:3px;">'
                '└ カバー元：記録なし（カバー判定の根拠は収集結果一覧を参照）</div>'
            )
            detail_rows += (
                f'<tr><td style="white-space:nowrap;color:#2E7D32;font-weight:700;">✅ カバー済み</td>'
                f'<td>{_to_ja_fields(_esc(i))}{covered_by_html}</td></tr>'
            )
    detail_table = (
        f'<details style="margin-top:8px;">'
        f'<summary style="cursor:pointer;font-size:12px;color:#1565C0;font-weight:700;">'
        f'🧮 論点別の判定を表示（全{len(issues)}件）</summary>'
        f'<table class="info-table" style="margin-top:8px;">'
        f'<tr><td style="width:15%;">判定</td><td>調査論点</td></tr>{detail_rows}</table>'
        f'</details>'
        if issues else ""
    )

    color = "#2E7D32" if (issues and not uncovered) else "#C62828"
    return (
        f'<h2 style="color:{color};border-left-color:{color};">🧮 網羅性チェック（調査論点のカバー状況）</h2>'
        f'<div style="font-size:12px;color:#666;margin-bottom:8px;">'
        f'調査開始前に洗い出した論点ごとに、対応する法令・情報を収集できたかを'
        f'AIの調査完了判断とは独立に検証した結果です。</div>'
        f'{summary_box}{detail_table}'
    )


def _build_law_matrix(law_items: list, excluded_laws: list, equipment_info: dict) -> str:
    """法令確認の全体像（確認漏れチェック用・常設表示）。

    1. ルールベース必須法令との突合：設備属性から機械的に導出した必須確認法令
       （AIの判断を介さない決定的リスト）が、最終的な法令リスト（該当・要確認・
       非該当）に反映されているかを突き合わせる。AIの論点出しに漏れがあっても
       ここで検出できる。
    2. 確認済み法令の全件一覧：該当・要確認・非該当を1つの表で示す。
       非該当が0件でもセクションを表示し、「どこまで確認したか」を常に明示する。
    """
    listed   = [(l.get("law_name", ""), l) for l in law_items if l.get("law_name")]
    excluded = [(e.get("law_name", ""), e) for e in excluded_laws if e.get("law_name")]

    # ── 1. ルールベース必須法令との突合 ──
    baseline = get_suggested_keywords(equipment_info or {})

    def _area_hit(base: str, names: list) -> str:
        """法令領域の照合（緩い部分一致）。通称・略称（フロン排出抑制法等）は
        正式法令名に解決してから照合する。施行令・施行規則が載っていれば
        その領域は確認済みとみなす。マッチした法令名を返す。"""
        candidates = {c for c in (base, resolve_law_alias(base)) if c}
        for name, _item in names:
            if name and any(c in name or name in c for c in candidates):
                return name
        return ""

    missing: list[str] = []
    base_rows = ""
    for base in baseline:
        hit = _area_hit(base, listed)
        if hit:
            status = '<span style="color:#2E7D32;font-weight:700;">✅ 掲載</span>'
            note = f'法令リストに掲載（{_esc(hit)}）'
        else:
            hit = _area_hit(base, excluded)
            if hit:
                status = '<span style="color:#555;font-weight:700;">🚫 非該当</span>'
                note = f'確認のうえ非該当と判断（{_esc(hit)}）'
            else:
                missing.append(base)
                status = '<span style="color:#B71C1C;font-weight:700;">⚠️ 未掲載</span>'
                note = ('<span style="color:#B71C1C;">最終リストに見当たりません。'
                        '適用要否を手動で確認してください</span>')
        base_rows += (
            f'<tr><td style="font-weight:600;">{_esc(base)}</td>'
            f'<td style="white-space:nowrap;">{status}</td><td>{note}</td></tr>'
        )

    if missing:
        base_summary = (
            f'<div style="background:#FFEBEE;border:1px solid #EF9A9A;border-left:5px solid #C62828;'
            f'border-radius:6px;padding:12px 18px;font-size:13px;margin-bottom:8px;">'
            f'<strong>🚨 突合結果 NG：</strong>設備属性から機械的に導出した必須確認法令 '
            f'{len(baseline)}件のうち、<strong>{len(missing)}件（{_esc("、".join(missing))}）が'
            f'最終リストに見当たりません。</strong>確認漏れの可能性があるため、適用要否を手動で確認してください。</div>'
        )
    else:
        base_summary = (
            f'<div style="background:#E8F5E9;border:1px solid #A5D6A7;border-left:5px solid #2E7D32;'
            f'border-radius:6px;padding:12px 18px;font-size:13px;color:#1B5E20;margin-bottom:8px;">'
            f'<strong>✅ 突合結果 OK：</strong>設備属性から機械的に導出した必須確認法令 '
            f'{len(baseline)}件は、すべて最終リスト（該当・要確認・非該当のいずれか）に反映されています。</div>'
        )

    base_table = (
        f'<details style="margin-bottom:16px;">'
        f'<summary style="cursor:pointer;font-size:12px;color:#1565C0;font-weight:700;">'
        f'🧾 必須確認法令との突合結果を表示（全{len(baseline)}件）</summary>'
        f'<table class="info-table" style="margin-top:8px;">'
        f'<tr><td style="width:32%;">必須確認法令（ルールベース）</td>'
        f'<td style="width:12%;">判定</td><td>反映先</td></tr>{base_rows}</table>'
        f'</details>'
    )

    # ── 2. 確認済み法令の全件一覧（該当・要確認・非該当） ──
    matrix_rows = ""
    _prio_rank = {"required": 0, "check": 1}
    for name, law in sorted(listed, key=lambda x: _prio_rank.get(x[1].get("priority", "check"), 1)):
        p = law.get("priority", "check")
        cfg = PRIORITY_CONFIG.get(p, PRIORITY_CONFIG["check"])
        n_del = len([d for d in law.get("deliveries", []) if d.get("item")])
        n_act = len([a for a in law.get("internal_actions", []) if a.get("item")])
        detail = "　".join(
            s for s in (
                f"届出・申請 {n_del}件" if n_del else "",
                f"社内対応 {n_act}件" if n_act else "",
            ) if s
        ) or "対応事項なし（適用要否の確認のみ）"
        matrix_rows += (
            f'<tr><td style="font-weight:600;">{_esc(name)}</td>'
            f'<td style="white-space:nowrap;"><span class="badge badge-{p}">{cfg["label"]}</span></td>'
            f'<td>{detail}</td></tr>'
        )
    for name, e in excluded:
        matrix_rows += (
            f'<tr style="background:#FAFAFA;color:#555;">'
            f'<td style="font-weight:600;">{_esc(name)}</td>'
            f'<td style="white-space:nowrap;">🚫 非該当</td>'
            f'<td>{_to_ja_fields(_esc(e.get("reason", "")))}</td></tr>'
        )

    excluded_note = "" if excluded else (
        '<div style="font-size:12px;color:#666;margin-top:6px;">'
        '※ 今回「非該当」と断定した法令はありません。判断材料が不足している法令は'
        '非該当とせず「要確認」に含めています（安全側の運用）。</div>'
    )

    return (
        f'<h2>🧾 法令確認の全体像（確認漏れチェック）</h2>'
        f'<div style="font-size:12px;color:#666;margin-bottom:8px;">'
        f'確認漏れがないことを検証するためのセクションです。上段は設備属性から'
        f'機械的に導出した必須確認法令（AIの判断を介さない決定的リスト）との突合結果、'
        f'下段は該当・要確認・非該当を含む確認済み法令の全件一覧です。'
        f'非該当の判断理由に誤りがないかもご確認ください。'
        f'前提となる設備情報が変わった場合は再調査が必要です。</div>'
        f'{base_summary}'
        f'{base_table}'
        f'<table class="info-table">'
        f'<tr><td style="width:32%;">法令名</td><td style="width:14%;">判定</td>'
        f'<td>対応事項／非該当の理由</td></tr>{matrix_rows}'
        f'</table>'
        f'{excluded_note}'
    )


def _build_unknown_items(items: list) -> str:
    if not items:
        return ""
    return "".join(
        f'<div class="unknown-item">❓ {_esc(item)}</div>' for item in items
    )


def _build_law_refs(search_results: list) -> str:
    refs = [r for r in search_results if r.get("title") and "error" not in r.get("source", "")]
    # Google検索グラウンディングの検索結果（タイトル・リンク・サマリー）は
    # Gemini API 追加利用規約により保存が制限されるため、恒久保存される
    # レポートには掲載せず、件数のみ注記する（結果確認画面では表示される）
    _grounding_sources = ("Gemini Grounding", "Gemini Summary")
    n_web = sum(1 for r in refs if r.get("source") in _grounding_sources)
    refs = [r for r in refs if r.get("source") not in _grounding_sources]
    # 社内文書はチャンク（抜粋）単位のヒットをファイル単位に集約する
    # （同一文書の抜粋が何行も並ぶ・関連の薄い文書が目立つのを防ぐ）
    internal = [r for r in refs if r.get("source") == "社内文書"]
    refs     = [r for r in refs if r.get("source") != "社内文書"]
    if not refs and not internal and not n_web:
        return ""
    html = ""
    for r in refs:
        url = r.get("url", "")
        title = _esc(r.get("title", ""))
        source = _esc(r.get("source", ""))
        link = f'<a href="{_esc(url)}" target="_blank">{title}</a>' if url else title
        html += f'<div class="law-ref">📖 {link} <span style="color:#888">({source})</span></div>'
    if internal:
        by_file: dict = {}
        for r in internal:
            m = re.match(r"社内文書:\s*(.+?)（抜粋", r.get("title", ""))
            fname = m.group(1).strip() if m else r.get("title", "")
            by_file[fname] = by_file.get(fname, 0) + 1
        for fname, n in by_file.items():
            html += (
                f'<div class="law-ref">📁 {_esc(fname)} '
                f'<span style="color:#888">(社内文書・ヒット抜粋 {n}件)</span></div>'
            )
        html += (
            '<div class="law-ref" style="color:#888;">'
            '※ 社内文書はベクトル検索でヒットした文書の一覧です。'
            '本件と関連の薄い文書が含まれる場合があります（採否はAIが判断済み）。</div>'
        )
    if n_web:
        html += (
            f'<div class="law-ref" style="color:#888;">🌐 このほか Web検索（Google検索）で '
            f'{n_web}件の公開情報を参照しました。検索結果のリンクは Google の利用規約により'
            f'本レポートには掲載していません（調査時の結果確認画面で参照できます）。</div>'
        )
    return html
