"""
e-Gov 法令 API v2 ラッパー
API仕様: https://laws.e-gov.go.jp/api/2/swagger-ui
"""

import re
import base64
import requests
from xml.etree import ElementTree as ET

EGOV_BASE_V2 = "https://laws.e-gov.go.jp/api/2"

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "LawCheckAI/1.0 (research; contact: legal-ai-dev)",
    "Accept": "application/json",
})


# 法令名の末尾に現れる典型的なサフィックス
_LAW_SUFFIXES = ("法", "令", "規則", "条例", "規程", "基準", "指針", "告示", "省令", "政令", "勅令")

# 通称・略称 → 正式法令名のマッピング
_LAW_ALIAS: dict[str, str] = {
    "フロン排出抑制法":        "フロン類の使用の合理化及び管理の適正化に関する法律",
    "化管法":                  "特定化学物質の環境への排出量の把握等及び管理の改善の促進に関する法律",
    "安衛法":                  "労働安全衛生法",
    "高圧ガス法":              "高圧ガス保安法",
    "毒劇法":                  "毒物及び劇物取締法",
    "廃掃法":                  "廃棄物の処理及び清掃に関する法律",
    "廃棄物処理法":            "廃棄物の処理及び清掃に関する法律",
    "ばい煙規制法":            "大気汚染防止法",
    "放射線障害防止法":        "放射性同位元素等の規制に関する法律",
    "消防法施行令":            "消防法施行令",
}


def resolve_law_alias(name: str) -> str:
    """通称・略称を正式法令名に解決する（未登録ならそのまま返す）。"""
    return _LAW_ALIAS.get(name, name)


def _extract_law_name_candidates(keyword: str) -> list[str]:
    """
    「大気汚染防止法 排気 届出」のような複合クエリから法令名候補を抽出する。
    スペースなし → そのまま返す。
    スペースあり → 先頭から順に結合して法令名サフィックスで終わる部分を候補とする。
    """
    if " " not in keyword:
        return [keyword]
    tokens = keyword.split()
    candidates: list[str] = []
    # 先頭から結合しながら法令名サフィックスを探す（最長優先）
    for length in range(len(tokens), 0, -1):
        candidate = "".join(tokens[:length])
        if any(candidate.endswith(s) for s in _LAW_SUFFIXES):
            candidates.append(candidate)
    # 先頭トークン単体も候補に追加（バックストップ）
    if tokens[0] not in candidates:
        candidates.append(tokens[0])
    return candidates


def search_laws_by_keyword(keyword: str, max_results: int = 5) -> list[dict]:
    # 通称・略称を正式名称に変換
    keyword = _LAW_ALIAS.get(keyword, keyword)

    """
    キーワードで法令を検索する。
    1. law_title 検索: キーワードそのまま → ヒットなければ法令名候補に分解してリトライ
    2. keyword 全文検索: 残件数を補完
    """
    results = []
    seen_ids: set[str] = set()

    def _add(hits: list[dict]) -> None:
        for law in hits:
            law_id = law.get("law_id", "")
            if law_id and law_id not in seen_ids:
                seen_ids.add(law_id)
                results.append(law)

    # ── 法令名検索（law_title）──────────────────────────────────────
    _add(_search_by_title(keyword, max_results))

    # ヒットなし かつ スペース含む → 法令名候補に分解してリトライ
    if not results and " " in keyword:
        for candidate in _extract_law_name_candidates(keyword):
            if candidate == keyword:
                continue
            _add(_search_by_title(candidate, max_results))
            if results:
                break  # 最初にヒットした候補で打ち止め

    # ── 全文検索（keyword）で補完 ───────────────────────────────────
    if len(results) < max_results:
        # スペース含む複合クエリは先頭の法令名候補だけで全文検索（ノイズ軽減）
        fulltext_query = _extract_law_name_candidates(keyword)[0] if " " in keyword else keyword
        _add(_search_by_fulltext(fulltext_query, (max_results - len(results)) * 2))

    return results[:max_results]


def _parse_law(law_dict: dict, keyword: str) -> dict:
    """APIレスポンスの1件分を統一フォーマットに変換する"""
    li = law_dict.get("law_info", {})
    ri = law_dict.get("revision_info", {}) or law_dict.get("current_revision_info", {})
    law_id = li.get("law_id", "")
    # e-Gov v2 では full text 取得に law_revision_id (UUID) が必要
    law_revision_id = (
        ri.get("law_revision_id", "")
        or ri.get("revision_id", "")
        or li.get("law_revision_id", "")
    )
    return {
        "source":           "e-Gov v2",
        "law_id":           law_id,
        "law_revision_id":  law_revision_id,
        "law_number":       li.get("law_num", ""),
        "title":            ri.get("law_title", ""),
        "category":         ri.get("category", ""),
        "last_amended":     ri.get("amendment_enforcement_date", ""),
        "url":              f"https://laws.e-gov.go.jp/law/{law_id}" if law_id else "",
        "keyword":          keyword,
    }


def _search_by_title(keyword: str, max_results: int) -> list[dict]:
    """GET /api/2/laws?law_title=... による法令名検索"""
    try:
        resp = _SESSION.get(
            f"{EGOV_BASE_V2}/laws",
            params={"law_title": keyword, "limit": max_results},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return [_parse_law(law, keyword) for law in data.get("laws", [])]
    except Exception as e:
        print(f"[e-Gov title search ERROR] '{keyword}': {e}")
        return []


def _search_by_fulltext(keyword: str, max_results: int) -> list[dict]:
    """GET /api/2/keyword?keyword=... による本文キーワード検索"""
    try:
        resp = _SESSION.get(
            f"{EGOV_BASE_V2}/keyword",
            params={"keyword": keyword, "limit": max_results},
            timeout=10,
        )
        if resp.status_code == 404:
            return []  # 全文検索非対応キーワードは静かにスキップ
        resp.raise_for_status()
        data = resp.json()
        return [_parse_law(item, keyword) for item in data.get("items", [])]
    except Exception as e:
        print(f"[e-Gov fulltext search ERROR] '{keyword}': {e}")
        return []


# ─── ルールベース法令チェックリスト（決定的ベースライン） ─────────
# 設備属性 → 必ず確認する法令のマトリクス。
# LLMの分析キーワードとマージして必ず検索し、LLMの発想漏れによる確認漏れを防ぐ。
# 値が「なし」と明言された場合のみ非該当。「不明」「未定」は安全側（該当しうる）に倒す。
EQUIPMENT_KEYWORD_MAP = {
    "chemicals": [
        "有機溶剤中毒予防規則",
        "特定化学物質障害予防規則",
        "危険物の規制に関する政令",
        "毒物及び劇物取締法",
        "高圧ガス保安法",
    ],
    "fire_exhaust": [
        "消防法",
        "大気汚染防止法",
        "粉じん障害防止規則",
    ],
    "wastewater": [
        "水質汚濁防止法",
        "下水道法",
        "廃棄物の処理及び清掃に関する法律",
    ],
    "noise_vibration": [
        "騒音規制法",
        "振動規制法",
    ],
    "radiation": [
        "放射性同位元素等の規制に関する法律",
        "電離放射線障害防止規則",
    ],
    "construction": [
        "建築基準法",
        "消防法",
        "電気事業法",
    ],
}

# 設備情報の記述内容（全項目の値）に対するキーワードトリガー
_VALUE_KEYWORD_TRIGGERS: list[tuple] = [
    (re.compile(r"冷媒|フロン|R\d{2,3}[A-Za-z]?|冷凍|チラー"),
     ["フロン排出抑制法", "冷凍保安規則", "高圧ガス保安法"]),
    (re.compile(r"ボイラー|圧力容器"),
     ["ボイラー及び圧力容器安全規則"]),
    (re.compile(r"クレーン|ホイスト|巻上げ"),
     ["クレーン等安全規則"]),
    (re.compile(r"粉じん|研磨|グラインダ|切削粉|サンドブラスト"),
     ["粉じん障害防止規則"]),
    (re.compile(r"燃料|灯油|軽油|ガソリン|重油|LPG|都市ガス"),
     ["消防法", "危険物の規制に関する政令"]),
]

# 常に確認する基本法令
_BASE_KEYWORDS = ["労働安全衛生法", "消防法"]

_NEGATIVE_VALUES = ("なし", "無し", "特になし")


def _is_potentially_applicable(val) -> bool:
    """「なし」と明言された場合のみ False。「不明」「未定」や具体的記載は
    安全側（該当しうる＝True）に倒す。空欄は判断材料がないため False。"""
    s = str(val or "").strip()
    if not s:
        return False
    return not any(
        s == n or s.startswith(n + "（") or s.startswith(n + "。")
        for n in _NEGATIVE_VALUES
    )


def get_suggested_keywords(equipment_info: dict) -> list[str]:
    """設備情報から必ず確認すべき法令キーワードを返す（ルールベース・決定的）。
    LLMの判断を介さないため、設備属性に対応する法令の検索が保証される。"""
    keywords = list(_BASE_KEYWORDS)
    for field, laws in EQUIPMENT_KEYWORD_MAP.items():
        if _is_potentially_applicable(equipment_info.get(field)):
            keywords.extend(laws)
    all_text = " ".join(str(v) for v in equipment_info.values())
    for pat, laws in _VALUE_KEYWORD_TRIGGERS:
        if pat.search(all_text):
            keywords.extend(laws)
    return list(dict.fromkeys(keywords))


# 法令XMLのインメモリキャッシュ（law_id → XML文字列）
_LAW_XML_CACHE: dict[str, str] = {}


def _get_law_xml(law_id: str, law_revision_id: str = "") -> str:
    """e-Gov API から法令XML文字列を取得する（キャッシュ付き）。失敗時は空文字。
    /law_data/{id} は law_full_text_format=xml 指定時、Base64エンコードされた
    XML文字列を返す。"""
    cache_key = law_revision_id or law_id
    xml_str = _LAW_XML_CACHE.get(cache_key)
    if xml_str:
        return xml_str

    # 試行するIDの優先順（revision_id → law_id）
    candidates = []
    if law_revision_id:
        candidates.append(law_revision_id)
    if law_id and law_id not in candidates:
        candidates.append(law_id)

    for try_id in candidates:
        try:
            resp = _SESSION.get(
                f"{EGOV_BASE_V2}/law_data/{try_id}",
                params={"law_full_text_format": "xml"},
                timeout=20,
            )
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            raw = resp.json().get("law_full_text", "")
            if isinstance(raw, str) and raw:
                if raw.lstrip().startswith("<"):
                    xml_str = raw          # 素のXMLで返ってきた場合
                else:
                    xml_str = base64.b64decode(raw).decode("utf-8")
            if xml_str:
                _LAW_XML_CACHE[cache_key] = xml_str
                return xml_str
        except Exception:
            continue
    return ""


def fetch_article_text(law_id: str, article_refs: list[str],
                       law_revision_id: str = "") -> dict[str, str]:
    """
    e-Gov API から指定法令の条文テキストを取得する。
    article_refs: ["第10条", "第11条第1項"] などの形式
    戻り値: {"第10条": "条文テキスト...", ...}  失敗時は {}（エラーキーなし）
    """
    if not law_id:
        return {}

    xml_str = _get_law_xml(law_id, law_revision_id)
    if not xml_str:
        return {}   # エラーではなく空を返す → 表示側でリンクを出す

    return _extract_articles(xml_str, article_refs)


def fetch_article_captions(law_id: str, article_refs: list[str],
                           law_revision_id: str = "") -> dict[str, str]:
    """
    指定条番号の条見出し（ArticleCaption、例：（建築確認））を取得する。
    戻り値: {"第6条": "（建築確認）", ...}  見出しが無い条・失敗時は含めない
    """
    if not law_id or not article_refs:
        return {}

    xml_str = _get_law_xml(law_id, law_revision_id)
    if not xml_str:
        return {}

    try:
        xml_clean = re.sub(r' xmlns[^"]*"[^"]*"', "", xml_str)
        root = ET.fromstring(xml_clean)
    except ET.ParseError:
        return {}

    # Num属性 → 条見出しのマップを一度だけ構築。
    # 附則（SupplProvision）にも同じ条番号があるため、最初の出現（本則）を優先する
    # （_find_article_text の最初マッチ採用と同じ挙動）
    caption_by_num: dict[str, str] = {}
    for article in root.iter("Article"):
        num = article.get("Num", "").lstrip("0") or article.get("num", "")
        cap = article.find("ArticleCaption")
        if num and num not in caption_by_num and cap is not None:
            cap_text = _elem_text(cap)
            if cap_text:
                caption_by_num[num] = cap_text

    results: dict[str, str] = {}
    for ref in article_refs:
        # 「第28条の2の3」のような多段の枝番も含めて Num 属性形式（28_2_3）に変換する
        art_match = re.search(r'第(\d+)条((?:の\d+)*)', ref)
        if not art_match:
            continue
        art_num = art_match.group(1) + "".join(
            f"_{n}" for n in re.findall(r"の(\d+)", art_match.group(2))
        )
        cap = caption_by_num.get(art_num)
        if cap:
            results[ref] = cap
    return results


def normalize_article_ref(ref: str) -> str:
    """条番号文字列を「第◯条」「第◯条の◯」「第◯条第◯項」形式に正規化する。
    LLM由来の見出し・説明語句（例：「第30条（危険又は有害な業務の調査等）」）は取り除く。
    条番号が読み取れない場合（「第○章」「条文番号不明」等）は空文字を返す。
    """
    m = re.search(r'第(\d+)条((?:の\d+)*)\s*(?:第(\d+)項)?', ref or "")
    if not m:
        return ""
    out = f"第{m.group(1)}条{m.group(2)}"
    if m.group(3):
        out += f"第{m.group(3)}項"
    return out


def article_sort_key(ref: str) -> list:
    """条番号の昇順ソート用キーを返す。
    「第6条」→ [6]、「第28条の2」→ [28, 2]、「第10条第1項」→ [10, 0, ..., 1]。
    条番号は法令の章立て順に振られているため、このキーで並べると
    章順・条順の表示になる。読み取れない場合は末尾に回す。
    """
    s = re.sub(r"（[^）]*）", "", ref or "").strip()
    m = re.match(r"第(\d+)条((?:の\d+)*)\s*(?:第(\d+)項)?", s)
    if not m:
        return [float("inf")]
    key = [int(m.group(1))] + [int(n) for n in re.findall(r"の(\d+)", m.group(2))]
    # 「の◯」の枝番なし（=0扱い）と項番号を末尾に付け、桁数を揃えて比較する
    key += [0] * (3 - len(key))
    key.append(int(m.group(3)) if m.group(3) else 0)
    return key


def fetch_article_list(law_id: str, law_revision_id: str = "") -> list[dict]:
    """法令XMLから本則の全条番号と見出しの一覧を取得する。
    戻り値: [{"ref": "第28条の2", "caption": "（危険性又は有害性等の調査）"}, ...]
    見出しが無い条は caption を空文字にする。失敗時は空リスト。
    附則の重複条番号は本則（最初の出現）を優先する。
    """
    if not law_id:
        return []
    xml_str = _get_law_xml(law_id, law_revision_id)
    if not xml_str:
        return []
    try:
        xml_clean = re.sub(r' xmlns[^"]*"[^"]*"', "", xml_str)
        root = ET.fromstring(xml_clean)
    except ET.ParseError:
        return []

    items: list[dict] = []
    seen: set[str] = set()
    for article in root.iter("Article"):
        num = article.get("Num", "").lstrip("0") or article.get("num", "")
        if not num or num in seen:
            continue
        seen.add(num)
        # Num属性 "28_2" → 「第28条の2」
        parts = num.split("_")
        if not parts[0].isdigit():
            continue
        ref = f"第{parts[0]}条" + "".join(f"の{p}" for p in parts[1:])
        cap = article.find("ArticleCaption")
        cap_text = _elem_text(cap) if cap is not None else ""
        items.append({"ref": ref, "caption": cap_text})
    return items


def fetch_article_chapters(law_id: str, law_revision_id: str = "") -> dict[str, str]:
    """条番号→その条が属する章タイトルのマップを取得する。
    戻り値: {"第6条": "第二章　特定工場等に関する規制", ...}
    章立てのない法令（短い政令・規則等）・取得失敗時は空 dict。
    同じ条番号が複数章に現れることはないが、念のため最初の出現を優先する。
    """
    if not law_id:
        return {}
    xml_str = _get_law_xml(law_id, law_revision_id)
    if not xml_str:
        return {}
    try:
        xml_clean = re.sub(r' xmlns[^"]*"[^"]*"', "", xml_str)
        root = ET.fromstring(xml_clean)
    except ET.ParseError:
        return {}

    chapter_by_ref: dict[str, str] = {}
    for chapter in root.iter("Chapter"):
        title_elem = chapter.find("ChapterTitle")
        title = _elem_text(title_elem) if title_elem is not None else ""
        if not title:
            continue
        for article in chapter.iter("Article"):
            num = article.get("Num", "").lstrip("0") or article.get("num", "")
            if not num:
                continue
            parts = num.split("_")
            if not parts[0].isdigit():
                continue
            ref = f"第{parts[0]}条" + "".join(f"の{p}" for p in parts[1:])
            chapter_by_ref.setdefault(ref, title)
    return chapter_by_ref


def _extract_articles(xml_str: str, article_refs: list[str]) -> dict[str, str]:
    """
    法令XML文字列から指定条番号の条文テキストを抽出する。
    """
    try:
        # XML名前空間を除去してパース
        xml_clean = re.sub(r' xmlns[^"]*"[^"]*"', "", xml_str)
        root = ET.fromstring(xml_clean)
    except ET.ParseError:
        # 解析失敗は「取得できなかった」扱いで空を返す（表示側でリンクを出す）
        return {}

    results: dict[str, str] = {}

    for ref in article_refs:
        # "第10条" → "10"、"第11条の2" → "11_2"、"第28条の2の3" → "28_2_3"
        # （XMLのNum属性形式）。"第10条第1項" → article="10", para="1"
        art_match = re.search(r'第(\d+)条((?:の\d+)*)', ref)
        para_match = re.search(r'第(\d+)項', ref)
        if not art_match:
            continue
        art_num = art_match.group(1) + "".join(
            f"_{n}" for n in re.findall(r"の(\d+)", art_match.group(2))
        )
        para_num = para_match.group(1) if para_match else None

        text = _find_article_text(root, art_num, para_num)
        if text:
            results[ref] = text

    return results


def _find_article_text(root: ET.Element, art_num: str, para_num: str | None) -> str:
    """XMLツリーから指定条・項のテキストを再帰的に探す。"""
    for article in root.iter("Article"):
        num = article.get("Num", "").lstrip("0") or article.get("num", "")
        if num != art_num:
            continue

        if para_num:
            # 指定した項だけ抽出
            for para in article.iter("Paragraph"):
                pnum = para.get("Num", "").lstrip("0") or para.get("num", "")
                if pnum == para_num:
                    return _article_text(para)
        else:
            return _article_text(article)

    return ""


def _elem_text(elem: ET.Element) -> str:
    """XML要素のテキストを再帰的に結合して返す。"""
    parts = []
    for node in elem.iter():
        if node.text and node.text.strip():
            parts.append(node.text.strip())
        if node.tail and node.tail.strip():
            parts.append(node.tail.strip())
    return "".join(parts)


# 条文整形: 番号ラベル要素（この後に全角スペースを入れる）
_LABEL_TAGS = {
    "ArticleTitle",     # 第五条
    "ParagraphNum",     # ２（第1項は空）
    "ItemTitle",        # 一
    "Subitem1Title",    # イ
    "Subitem2Title",
    "Subitem3Title",
}
# 条文整形: ブロック要素（この前で改行する）
_BLOCK_TAGS = {"Paragraph", "Item", "Subitem1", "Subitem2", "Subitem3"}


def _article_text(elem: ET.Element) -> str:
    """条・項のXMLを、法令の標準的な組版に合わせて整形したテキストで返す。
    - 条見出し（…）の後は改行
    - 条名「第五条」・項番号「２」・号番号「一」の後は全角スペース
    - 項・号・イロハの区切りは改行
    例:
        （指定製品及び特定製品の管理者の責務）
        第五条　指定製品の管理者は、…
        ２　特定製品の管理者は、…
    """
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        # ブロックの前で改行する。ただし直前が「第五条　」等の番号ラベルなら
        # 同じ行に続ける（第1項は条名と同一行に書くのが法令の組版）
        if node.tag in _BLOCK_TAGS and parts and not parts[-1].endswith(("　", "\n")):
            parts.append("\n")
        if node.tag == "ArticleCaption":
            t = _elem_text(node)
            if t:
                parts.append(t + "\n")
            return
        if node.tag in _LABEL_TAGS:
            t = _elem_text(node)
            if t:
                parts.append(t + "　")
            return
        if node.text and node.text.strip():
            parts.append(node.text.strip())
        for child in node:
            walk(child)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())

    walk(elem)
    return "".join(parts).strip()
