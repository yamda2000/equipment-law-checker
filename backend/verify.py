"""調査結果の対応事項（届出・申請）をAIで再チェックする（読み取り専用）。

各届出について、収集済みの根拠（e-Gov条文抜粋・Web出典・設備情報）に照らして
「根拠あり(ok) / 要確認(check) / 矛盾の疑い(conflict)」を判定し、理由を返す。
ワークフローの state・checkpointer は一切変更しない。判定はあくまでAIの補助であり、
最終確認は担当者・所轄窓口が行う前提。
"""

import logging

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

_MAX_ARTICLE_CHARS = 8000
_MAX_WEB_CHARS = 3000
_MAX_EQUIP_CHARS = 2000


VERIFY_SYSTEM = """あなたは設備導入時の法令確認レポートの品質チェック担当です。
提示された「届出・申請事項」を1件ずつ、一緒に提示する根拠（e-Gov法令の条文抜粋・
Web出典・設備情報）と照らして、内容の妥当性を判定してください。

## 判定区分（verdict）
- ok: 提示された根拠（条文抜粋・Web出典）で、その届出の要否・届出先・期限・根拠条文が
      妥当と確認できる
- check: 提示された根拠の中に確認材料が見当たらない、または情報不足で妥当性を確認できない
        （「間違い」という意味ではなく「この情報だけでは裏付けられない＝要確認」）
- conflict: 提示された根拠と矛盾する疑いがある（例：条文の数量閾値と設備の数量が合わない、
           届出先が条文・出典と異なる、期限が根拠と異なる など）

## ルール
- **判定は提示された根拠に基づく**こと。あなたの記憶だけで「正しい」と断定しない。
  根拠に裏付けが無ければ ok ではなく check にする。
- reason は1〜2文で、何を根拠にどう判断したかを具体的に書く
  （例：「消防法施行令第◯条の指定数量5分の1以上に該当し妥当」
   「届出先の根拠が条文・出典に見当たらず要確認」）。
- 条例など条文抜粋が無い項目は、Web出典で裏付けが取れなければ check にする。
- 断定的・安心させる表現は避け、確認を促す姿勢で書く。
- 入力の各届出には「【N】」の番号が付いている。index にはその N をそのまま返す。
"""


class ItemCheck(BaseModel):
    index: int = Field(description="対象の届出の番号（入力の「【N】」の N をそのまま）")
    verdict: str = Field(description="ok / check / conflict のいずれか")
    reason: str = Field(description="判定理由（1〜2文）")


class LawCheckResult(BaseModel):
    checks: list[ItemCheck] = Field(default_factory=list)


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…（省略）"


def verify_law_deliveries(
    law: dict,
    article_texts: dict | None = None,
    web_context: str = "",
    equip_summary: str = "",
    config: dict | None = None,
) -> dict:
    """1法令の deliveries を判定し {届出index: {"verdict","reason"}} を返す。

    article_texts: {条番号: 条文テキスト}（この法令の e-Gov 条文抜粋。条例では空）
    web_context:   届出先の根拠・Web出典スニペットなどの補助テキスト
    equip_summary: 設備情報の要約（数量閾値の照合などに使う）
    失敗時は空 dict（呼び出し側で「チェックできず」表示にする）。
    """
    from backend.workflow import _llm  # 循環importを避けるため遅延import

    deliveries = law.get("deliveries", []) or []
    if not deliveries:
        return {}

    law_name = law.get("law_name", "")
    art_block = "\n\n".join(
        f"{ref}\n{text}" for ref, text in (article_texts or {}).items() if text
    )
    art_block = _clip(art_block, _MAX_ARTICLE_CHARS) or "（この法令の条文抜粋は取得できていません）"

    deliveries_block = "\n".join(
        f"【{i}】 届出名: {d.get('item', '')}\n"
        f"　届出先: {d.get('authority', '') or '（未記載）'}"
        f"／期限: {d.get('deadline', '') or '（未記載）'}"
        f"／優先度: {d.get('priority', '')}"
        f"／根拠条文: {d.get('law_article', '') or '（未特定）'}\n"
        f"　届出先の根拠: {d.get('authority_basis', '') or '（記載なし）'}"
        for i, d in enumerate(deliveries)
    )

    ctx = (
        f"# 対象法令\n{law_name}\n\n"
        f"# この設備への適用理由\n{law.get('applicability', '')}\n\n"
        f"# 設備情報（要約）\n{_clip(equip_summary, _MAX_EQUIP_CHARS) or '（なし）'}\n\n"
        f"# 判定対象の届出・申請事項\n{deliveries_block}\n\n"
        f"# 根拠1：e-Gov条文抜粋\n{art_block}\n\n"
        f"# 根拠2：Web出典・届出先の根拠\n{_clip(web_context, _MAX_WEB_CHARS) or '（なし）'}"
    )

    try:
        res: LawCheckResult = _llm().with_structured_output(LawCheckResult).invoke(
            [SystemMessage(VERIFY_SYSTEM), HumanMessage(ctx)],
            config=config or {},
        )
    except Exception:
        logger.warning("対応事項の検証LLM呼び出しに失敗 '%s'", law_name, exc_info=True)
        return {}

    out: dict = {}
    for c in res.checks:
        v = str(c.verdict or "").strip().lower()
        if v not in ("ok", "check", "conflict"):
            v = "check"
        if 0 <= c.index < len(deliveries):
            out[c.index] = {"verdict": v, "reason": (c.reason or "").strip()}
    return out
