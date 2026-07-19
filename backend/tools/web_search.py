"""
Web検索ツール
google-genai SDK (v2) + Google Search Grounding を使用する。
GEMINI_API_KEY が未設定の場合はスタブデータを返す。
429 レート制限時は指数バックオフで最大3回リトライする。

注意: Grounding が返す URL へのアクセス（ページ本文のスクレイピング）は
Gemini API 追加利用規約で禁止されているため行わない。
"""

import os
import time

from backend.observability import observe_web_search


_MAX_RETRIES = 3
_RETRY_BASE_WAIT = 5  # 秒（5 → 10 → 20 と増加）

# Gemini Web検索の累積使用量（プロセス単位）。
# アプリ側（app.py）がセッション開始時点との差分を読み取ってコスト表示に使う。
_GEMINI_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}


def get_gemini_usage() -> dict:
    """Gemini Web検索の累積使用量スナップショットを返す。"""
    return dict(_GEMINI_USAGE)


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
            # Langfuse に generation として記録（未設定時は no-op）
            with observe_web_search(model_name, safe_query) as obs:
                response = client.models.generate_content(
                    model=model_name,
                    contents=safe_query,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        response_modalities=["TEXT"],
                    ),
                )

                # 使用量を積算（コスト表示用）
                _GEMINI_USAGE["requests"] += 1
                usage = getattr(response, "usage_metadata", None)
                prompt_tokens = completion_tokens = 0
                if usage is not None:
                    prompt_tokens     = getattr(usage, "prompt_token_count", 0) or 0
                    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
                    _GEMINI_USAGE["prompt_tokens"]     += prompt_tokens
                    _GEMINI_USAGE["completion_tokens"] += completion_tokens
                obs.record(response.text or "", prompt_tokens, completion_tokens)

            results = []
            seen_urls: set[str] = set()

            for candidate in response.candidates or []:
                grounding_meta = getattr(candidate, "grounding_metadata", None)
                if not grounding_meta:
                    continue

                # 検索候補（searchEntryPoint）。Gemini API 追加利用規約により、
                # Grounding 結果を表示する際は検索候補もあわせてエンドユーザーに
                # 表示する必要がある。呼び出し側で検索結果とは分離して扱う。
                sep = getattr(grounding_meta, "search_entry_point", None)
                rendered = getattr(sep, "rendered_content", "") if sep else ""
                if rendered:
                    results.append({
                        "source": "SearchSuggestions",
                        "title": "",
                        "url": "",
                        "snippet": "",
                        "query": safe_query,
                        "suggestions_html": rendered,
                    })

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
    検索対象を行政の公式サイトに限定する site: フィルタを付与する。

    注意: クエリ本文はそのまま外部（Gemini API）に送信される（内容の
    マスキングは行っていない）。Gemini API 無料枠では送信内容が Google の
    プロダクト改善に利用され得るため、機密情報をクエリに含めないこと。
    機密性の高い運用では有料枠（Cloud 請求先アカウント有効化）を使用する。
    """
    official_sites = (
        "site:city.yokohama.lg.jp OR site:www.pref.kanagawa.jp OR "
        "site:www.mhlw.go.jp OR site:www.fdma.go.jp OR site:www.env.go.jp OR "
        "site:www.mlit.go.jp OR site:www.meti.go.jp OR site:laws.e-gov.go.jp"
    )
    return f"{query} ({official_sites})"


def _stub_results(query: str) -> list[dict]:
    """GEMINI_API_KEY 未設定時の明示的な「未実行」結果。

    検索結果として扱ってはならない（実結果と混同すると、網羅性チェックが
    このプレースホルダーを根拠に論点カバー済みと誤判定する）。呼び出し側は
    source == "unavailable" を実結果から除外し、未実行として表示すること。
    """
    return [
        {
            "source": "unavailable",
            "title": f"[Web検索未実行] {query}",
            "url": "",
            "snippet": "GEMINI_API_KEY が未設定のためWeb検索は実行されていません。"
                       "この論点のWeb情報（条例・届出先等）は未確認です。",
            "query": query,
        }
    ]
