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
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar

from backend.observability import observe_web_search

logger = logging.getLogger(__name__)


_MAX_RETRIES = 3
_RETRY_BASE_WAIT = 5  # 秒（5 → 10 → 20 と増加）
# 1リクエストのタイムアウト（ミリ秒）。Streamlit は同期実行のため、
# 応答が返らないと調査フェーズ全体が固まる。必ず上限を設ける
_TIMEOUT_MS = int(os.getenv("GEMINI_WEB_SEARCH_TIMEOUT_MS", "60000"))

# Gemini Web検索の累積使用量（プロセス単位・運用監視用）。
# ※ セッション別のコスト表示には使わないこと。1プロセスが複数セッションを
#    捌くため、この値の差分を取ると他セッションの使用分まで混ざる（誤配賦）。
#    セッション単位の集計には collect_gemini_usage() を使う。
_GEMINI_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
_USAGE_LOCK = threading.Lock()

# 実行中のセッション（コンテキスト）ごとの集計先。
# collect_gemini_usage() のブロック内で実行された検索だけがここに積まれる。
_USAGE_SINK: ContextVar = ContextVar("gemini_usage_sink", default=None)


def get_gemini_usage() -> dict:
    """Gemini Web検索のプロセス累積使用量スナップショット（運用監視用）。"""
    with _USAGE_LOCK:
        return dict(_GEMINI_USAGE)


@contextmanager
def collect_gemini_usage():
    """このブロック内で実行した Web 検索の使用量だけを集計する。

    プロセス共有カウンタの差分方式では、複数セッションが同時に使うと
    他セッションの検索分まで自分のコストとして計上され（かつ各セッションが
    同じ分を計上するため二重計上になる）、正しく配賦できない。
    呼び出し側はこのブロックの戻り値だけを自セッションの使用量として扱う。
    """
    sink = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    token = _USAGE_SINK.set(sink)
    try:
        yield sink
    finally:
        _USAGE_SINK.reset(token)


def _record_usage(prompt_tokens: int, completion_tokens: int) -> None:
    """1回分の使用量を、プロセス累積と実行中セッションの集計先へ記録する。"""
    with _USAGE_LOCK:
        _GEMINI_USAGE["requests"]          += 1
        _GEMINI_USAGE["prompt_tokens"]     += prompt_tokens
        _GEMINI_USAGE["completion_tokens"] += completion_tokens
    sink = _USAGE_SINK.get()
    if sink is not None:
        sink["requests"]          += 1
        sink["prompt_tokens"]     += prompt_tokens
        sink["completion_tokens"] += completion_tokens


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


def _is_retryable(e: Exception) -> bool:
    """一時的な障害（再試行する価値がある）か判定する。

    レート制限に加えてタイムアウト・接続断・5xx も対象にする。
    認証不正や不正リクエストは何度試しても同じため対象外。
    """
    name = type(e).__name__.lower()
    if "timeout" in name or "connect" in name:
        return True
    s = str(e)
    return any(
        k in s for k in
        ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504",
         "UNAVAILABLE", "DEADLINE_EXCEEDED", "timed out", "Timeout")
    )


def _search_with_gemini(query: str, api_key: str) -> list[dict]:
    """google-genai SDK v2 + Google Search Grounding で検索（429時はリトライ）"""
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=_TIMEOUT_MS),
    )
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

                # 使用量を積算（プロセス累積＋実行中セッション分）
                usage = getattr(response, "usage_metadata", None)
                prompt_tokens = completion_tokens = 0
                if usage is not None:
                    prompt_tokens     = getattr(usage, "prompt_token_count", 0) or 0
                    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
                _record_usage(prompt_tokens, completion_tokens)
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
            if attempt < _MAX_RETRIES - 1 and _is_retryable(e):
                wait = _RETRY_BASE_WAIT * (2 ** attempt)
                logger.warning(
                    "Web検索が一時エラー（%s）。%d秒待機して再試行 (%d/%d)",
                    type(e).__name__, wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)
                continue
            # 恒久的なエラー（認証不正・不正リクエスト等）は即座に返す
            logger.warning("Web検索に失敗: %s", e)
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
