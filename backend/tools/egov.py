"""
e-Gov 法令 API v2 ラッパー
API仕様: https://laws.e-gov.go.jp/api/2/swagger-ui
"""

import re
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


# ─── キーワードマッピング ──────────────────────────────────────────
EQUIPMENT_KEYWORD_MAP = {
    "chemicals_あり": [
        "有機溶剤中毒予防規則",
        "特定化学物質障害予防規則",
        "危険物の規制に関する政令",
        "化学物質等の危険性又は有害性等の表示又は通知等の促進に関する指針",
    ],
    "fire_exhaust_あり": [
        "大気汚染防止法",
        "揮発性有機化合物の排出抑制に関する法律",
        "消防法",
    ],
    "wastewater_あり": [
        "水質汚濁防止法",
        "下水道法",
    ],
    "radiation_あり": [
        "放射線障害防止法",
        "医療法",
        "労働安全衛生法",
    ],
    "construction_あり": [
        "建築基準法",
        "消防法",
    ],
}


def get_suggested_keywords(equipment_info: dict) -> list[str]:
    """設備情報から推奨検索キーワードを返す"""
    keywords = set(["労働安全衛生法", "消防法"])
    for field, candidates in EQUIPMENT_KEYWORD_MAP.items():
        key, val = field.rsplit("_", 1)
        if val in str(equipment_info.get(key, "")):
            keywords.update(candidates)
    return list(keywords)


# 法令XMLのインメモリキャッシュ（law_id → XML文字列）
_LAW_XML_CACHE: dict[str, str] = {}


def fetch_article_text(law_id: str, article_refs: list[str],
                       law_revision_id: str = "") -> dict[str, str]:
    """
    e-Gov API から指定法令の条文テキストを取得する。
    article_refs: ["第10条", "第11条第1項"] などの形式
    戻り値: {"第10条": "条文テキスト...", ...}  失敗時は {}（エラーキーなし）
    """
    if not law_id:
        return {}

    cache_key = law_revision_id or law_id
    xml_str = _LAW_XML_CACHE.get(cache_key)
    if not xml_str:
        # 試行するエンドポイントIDの優先順（revision_id → law_id → law_id plural）
        candidates = []
        if law_revision_id:
            candidates.append(law_revision_id)
        candidates.append(law_id)

        for try_id in candidates:
            for path in [f"{EGOV_BASE_V2}/law/{try_id}",
                         f"{EGOV_BASE_V2}/laws/{try_id}"]:
                try:
                    resp = _SESSION.get(path, timeout=15)
                    if resp.status_code == 404:
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    xml_str = (
                        data.get("law_full_text", {}).get("xml", "")
                        or data.get("law", {}).get("law_body", {}).get("xml", "")
                        or data.get("xml", "")
                        or ""
                    )
                    if xml_str:
                        _LAW_XML_CACHE[cache_key] = xml_str
                        break
                except Exception:
                    continue
            if xml_str:
                break

    if not xml_str:
        return {}   # エラーではなく空を返す → 表示側でリンクを出す

    return _extract_articles(xml_str, article_refs)


def _extract_articles(xml_str: str, article_refs: list[str]) -> dict[str, str]:
    """
    法令XML文字列から指定条番号の条文テキストを抽出する。
    """
    try:
        # XML名前空間を除去してパース
        xml_clean = re.sub(r' xmlns[^"]*"[^"]*"', "", xml_str)
        root = ET.fromstring(xml_clean)
    except ET.ParseError:
        return {"error": "XML解析エラー"}

    results: dict[str, str] = {}

    for ref in article_refs:
        # "第10条" → "10"、"第10条第1項" → article="10", para="1"
        art_match = re.search(r'第(\d+)条', ref)
        para_match = re.search(r'第(\d+)項', ref)
        if not art_match:
            continue
        art_num = art_match.group(1)
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
                    return _elem_text(para)
        else:
            return _elem_text(article)

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
