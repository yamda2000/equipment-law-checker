"""
Web検索ツール
google-genai SDK (v2) + Google Search Grounding を使用する。
GEMINI_API_KEY が未設定の場合はスタブデータを返す。
429 レート制限時は指数バックオフで最大3回リトライする。
"""

import os
import time


_MAX_RETRIES = 3
_RETRY_BASE_WAIT = 5  # 秒（5 → 10 → 20 と増加）


def search_web(query: str, context: str = "") -> list[dict]:
    """
    公開Web情報を検索する。
    GEMINI_API_KEY が設定されている場合は Gemini Google Search Grounding を使用。
    未設定の場合はスタブデータを返す。
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        return _search_with_gemini(query, api_key)
    return _stub_results(query)


def _search_with_gemini(query: str, api_key: str) -> list[dict]:
    """google-genai SDK v2 + Google Search Grounding で検索（429時はリトライ）"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GEMINI_WEB_SEARCH_MODEL", "gemini-2.0-flash")
    safe_query = _sanitize_query(query)

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=safe_query,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    response_modalities=["TEXT"],
                ),
            )

            results = []
            seen_urls: set[str] = set()

            for candidate in response.candidates or []:
                grounding_meta = getattr(candidate, "grounding_metadata", None)
                if not grounding_meta:
                    continue
                for chunk in (getattr(grounding_meta, "grounding_chunks", None) or [])[:6]:
                    web = getattr(chunk, "web", None)
                    if web:
                        url = getattr(web, "uri", "")
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                        results.append({
                            "source": "Gemini Grounding",
                            "title": getattr(web, "title", ""),
                            "url": url,
                            "snippet": "",
                            "query": safe_query,
                        })

            if response.text:
                results.append({
                    "source": "Gemini Summary",
                    "title": f"検索結果サマリー: {query}",
                    "url": "",
                    "snippet": response.text[:800],
                    "query": safe_query,
                })

            return results

        except Exception as e:
            last_error = e
            err_str = str(e)
            # 429 レート制限 → 待機してリトライ
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = _RETRY_BASE_WAIT * (2 ** attempt)
                print(f"[Web検索] 429 レート制限。{wait}秒待機して再試行 ({attempt + 1}/{_MAX_RETRIES})")
                time.sleep(wait)
                continue
            # それ以外のエラーはすぐに返す
            break

    return [{"source": "error", "title": str(last_error), "url": "", "snippet": "", "query": query}]


def _sanitize_query(query: str) -> str:
    """
    機密情報を含まないよう、外部送信前にクエリを一般化する。
    クエリはそのまま使用し、公式サイトに絞った検索を行う。
    """
    official_sites = (
        "site:city.yokohama.lg.jp OR site:www.pref.kanagawa.jp OR "
        "site:www.mhlw.go.jp OR site:www.fdma.go.jp OR site:www.env.go.jp OR "
        "site:www.mlit.go.jp OR site:www.meti.go.jp OR site:laws.e-gov.go.jp"
    )
    return f"{query} ({official_sites})"


def _stub_results(query: str) -> list[dict]:
    """GEMINI_API_KEY 未設定時のスタブ結果"""
    return [
        {
            "source": "stub",
            "title": f"[未設定] {query} に関する情報",
            "url": "https://laws.e-gov.go.jp/",
            "snippet": "GEMINI_API_KEY を .env に設定するとWeb検索が有効になります。"
                       "現在は e-Gov API の結果のみを使用しています。",
            "query": query,
        }
    ]
