"""Langfuse 連携（LLMトレース・コスト計測）

LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY が設定されている場合のみ有効になる。
未設定・SDK未インストール・接続不可のいずれでも、アプリ本体の動作は変えず
すべて no-op にフォールバックする（観測機能の障害で業務フローを止めない）。

- LangGraph / LangChain の LLM 呼び出し: CallbackHandler を config に渡して自動トレース。
  モデル名（gpt-4o 等）から Langfuse 側がコストを自動計算する。
- Gemini Web検索（google-genai 直接呼び出し）: observe_web_search() で
  generation として手動記録し、トークン使用量からコストを計算させる。
- トレースは thread_id を session_id にして案件（スレッド）単位でグルーピングする。

## Google 検索グラウンディング結果の保存制限（Gemini API 追加利用規約）
Grounding 付き検索結果・検索候補（タイトル・リンク含む）は、規約で保存が
原則禁止されている（例外は表示評価目的30日等に限定）。Langfuse は保持期間が
これに収まらないため、Grounding 由来のテキストはトレースに記録しない：
- observe_web_search: 出力テキストは記録せず、トークン数・文字数のみ記録
- mask 関数: LLM プロンプト・LangGraph state 内の Grounding 由来のタイトル・
  スニペット・検索候補HTMLを送信前に [REDACTED] に置換

※ langfuse SDK v4 の API（start_as_current_observation / propagate_attributes / mask）を使用。
"""

import os
import re
import logging
from contextlib import contextmanager, ExitStack

logger = logging.getLogger(__name__)

_handler = None
_handler_failed = False
_client_initialized = False

# ─── Grounding 結果のマスキング（Gemini API 追加利用規約対応） ─────
_GROUNDING_SOURCES = ("Gemini Grounding", "Gemini Summary", "SearchSuggestions")
_REDACTED = "[REDACTED: Google検索グラウンディング結果（規約の保存制限により記録対象外）]"

# プロンプト文字列中の Grounding 由来エントリ：
# 「- タイトル (Gemini Grounding)」等のタグ付き行と、それに続く継続行
# （概要スニペット等。次の「- 」項目・「#」見出しで停止）をまとめて置換する
_GROUNDING_BLOCK_PAT = re.compile(
    r"^[^\S\n]*-[^\n]*\((?:Gemini Grounding|Gemini Summary)\)"
    r"(?:\n(?![^\S\n]*-|#)[^\n]*)*",
    re.MULTILINE,
)


def _mask_value(v):
    """トレース送信データから Grounding 由来のテキストを再帰的に除去する。"""
    if isinstance(v, str):
        return _GROUNDING_BLOCK_PAT.sub(_REDACTED, v)
    if isinstance(v, dict):
        # 検索結果 dict（source タグ付き）はタイトル・スニペット・URL を除去
        if str(v.get("source", "")) in _GROUNDING_SOURCES:
            masked = dict(v)
            for k in ("title", "snippet", "url", "suggestions_html", "html"):
                if masked.get(k):
                    masked[k] = _REDACTED
            return masked
        # 検索候補（state 内は {"query", "html"} 形式で source タグを持たない）
        if "suggestions_html" in v or ("html" in v and "query" in v):
            masked = dict(v)
            for k in ("suggestions_html", "html"):
                if masked.get(k):
                    masked[k] = _REDACTED
            return masked
        return {k: _mask_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_mask_value(x) for x in v]
    return v


def _mask_grounding(data, **kwargs):
    """Langfuse の mask フック。失敗時は安全側（全置換）に倒す。"""
    try:
        return _mask_value(data)
    except Exception:
        return _REDACTED


def langfuse_enabled() -> bool:
    """Langfuse のキーが設定されているか（実際に接続可能かまでは見ない）"""
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )


def _ensure_client() -> bool:
    """Grounding マスク関数付きで Langfuse クライアントを初期化する（初回のみ）。
    CallbackHandler や get_client() が mask なしのクライアントを暗黙生成する前に、
    必ずここを通すこと（シングルトンは最初の初期化の設定が使われる）。"""
    global _client_initialized
    if _client_initialized:
        return True
    try:
        from langfuse import Langfuse
        Langfuse(mask=_mask_grounding)
        _client_initialized = True
        return True
    except Exception:
        logger.exception("Langfuse クライアントの初期化に失敗（トレース無効で継続）")
        return False


def get_langchain_handler():
    """LangChain/LangGraph 用の Langfuse CallbackHandler（無効時は None）"""
    global _handler, _handler_failed
    if not langfuse_enabled() or _handler_failed:
        return None
    if _handler is None:
        try:
            if not _ensure_client():
                _handler_failed = True
                return None
            from langfuse.langchain import CallbackHandler
            _handler = CallbackHandler()
        except Exception:
            logger.exception("Langfuse CallbackHandler の初期化に失敗（トレース無効で継続）")
            _handler_failed = True
            return None
    return _handler


def _get_client():
    try:
        if not _ensure_client():
            return None
        from langfuse import get_client
        return get_client()
    except Exception:
        return None


@contextmanager
def trace_run(name: str, *, session_id: str = "", user_id: str = "", input=None):
    """ワークフロー実行1回分を1トレースとして記録するコンテキストマネージャ。

    yield 値は LangGraph の config にマージする dict
    （有効時: {"callbacks": [handler]} / 無効時: {}）。
    トレース名はルートスパン名、session_id は propagate_attributes で
    配下の全スパン（LangChain・Gemini手動計測とも）に伝播する。

    使い方:
        with trace_run("hearing", session_id=thread_id, input=text) as lf:
            workflow.invoke(..., config={**get_config(), **lf})
    """
    handler = get_langchain_handler()
    if handler is None:
        yield {}
        return

    lf_config = {"callbacks": [handler]}
    client = _get_client()
    if client is None:
        yield lf_config
        return

    stack = ExitStack()
    try:
        from langfuse import propagate_attributes
        stack.enter_context(propagate_attributes(
            session_id=session_id or None,
            user_id=user_id or None,
        ))
        stack.enter_context(client.start_as_current_observation(
            name=name, as_type="span", input=input,
        ))
    except Exception:
        logger.exception("Langfuse トレース開始に失敗（トレース無効で継続）")
        stack.close()
        yield lf_config
        return

    with stack:
        yield lf_config


class _WebSearchObservation:
    """observe_web_search が yield する記録用オブジェクト（無効時は no-op）"""

    def __init__(self, generation):
        self._gen = generation

    def record(self, output: str, input_tokens: int, output_tokens: int) -> None:
        if self._gen is None:
            return
        try:
            # 出力テキスト（Grounding 付き検索結果）は Gemini API 追加利用規約の
            # 保存制限により記録しない。コスト計算に必要なトークン数と、
            # 参考情報として文字数のみを記録する
            self._gen.update(
                output=f"{_REDACTED}（{len(output or '')}文字）",
                usage_details={
                    "input":  int(input_tokens or 0),
                    "output": int(output_tokens or 0),
                },
            )
        except Exception:
            pass


@contextmanager
def observe_web_search(model: str, query: str):
    """Gemini Web検索1回を Langfuse の generation として記録する。

    google-genai SDK の直接呼び出しは LangChain の callback に乗らないため、
    ここで手動計測する。モデル名とトークン使用量から Langfuse がコストを計算する。
    """
    gen = None
    if langfuse_enabled():
        client = _get_client()
        if client is not None:
            try:
                gen = client.start_observation(
                    name="gemini-web-search", as_type="generation",
                    model=model, input=query,
                )
            except Exception:
                gen = None
    try:
        yield _WebSearchObservation(gen)
    except Exception as e:
        if gen is not None:
            try:
                gen.update(level="ERROR", status_message=str(e)[:500])
            except Exception:
                pass
        raise
    finally:
        if gen is not None:
            try:
                gen.end()
            except Exception:
                pass
