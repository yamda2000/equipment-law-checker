"""案件コンテキストに基づく QA チェーン

確認フェーズ（調査方針確認・結果確認・レポート確認・完了後）で、担当者の
「わからないこと」にその場で回答する。LangGraph 本体の state・checkpointer
には一切触れない読み取り専用の一発呼び出しで、質問してもワークフローは
進まない（承認・再調査などの状態遷移は既存のボタン操作のみ）。
"""

import json
import logging
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.workflow import _llm

logger = logging.getLogger(__name__)

# law_items 等が大きい案件でもプロンプトが際限なく膨らまないための上限
_MAX_SECTION_CHARS = 15000
_MAX_HISTORY_CHARS = 4000
# 添付資料はカタログPDF等の大きめの資料も読めるよう別枠で広めに取る
_MAX_DOC_CHARS = 40000          # 1ファイルあたり
_MAX_DOC_SECTION_CHARS = 80000  # 添付資料セクション全体

QA_SYSTEM = """あなたは「設備導入時 法令・届出施設確認サポートAI」の補助アシスタントです。
担当者（法令に詳しくない場合が多い）が、調査の各段階（ヒアリング・調査方針確認・
調査結果確認・レポート確認）でわからないことを質問してきます。
提供された案件情報と一般的な法令知識に基づいて回答してください。

## 回答のルール
- 専門用語（特定施設・危険物・届出など）は、初めての人にもわかる言葉で説明する。
- この案件の設備情報に即して「あなたの設備の場合は…」と具体的に答える。
- ヒアリング中の質問には、AIの質問の意図・用語の意味・何をどう答えればよいか
  （単位、社内のどこ・誰に確認すればよいか等）を説明する。わからない項目は
  「不明」「確認中」と回答してよいことも伝える。ヒアリングへの回答自体は
  画面のヒアリング入力欄から送信するよう案内する。
- 調査結果（対象法令・優先度・届出・社内対応・非該当判断）への質問には、提供された
  「調査結果」「調査で参照した情報源」に基づいて説明する。情報源はタイトルの一覧のみ
  提供されるため、個別記事の本文内容までは断定しない。
- あなたは調査を行わない。調査結果の内容（対象法令・優先度・対応事項）を訂正・変更・追加しない。
- 調査結果に不足や誤りの疑いがある場合は、内容には踏み込まず「画面の『修正・追加調査を依頼する』
  （調査方針確認では『調査前に追記する』）ボタンから依頼してください」と案内する。
- 「質問時のWeb検索結果」が提供された場合、その内容も根拠にして回答してよい。
  使った場合は出典（検索結果のタイトル）を回答に添える。検索結果の概要から
  断定できないことは断定しない。
- 「質問時の社内文書検索結果」が提供された場合、その内容も根拠にして回答してよい。
  使った場合は出典（社内文書名）を回答に添える。抜粋のみのため断定はしない。
- 「質問時の法令検索結果（e-Gov）」が提供された場合、関連しうる法令の存在確認や
  出典リンクの提示に使ってよい。ただし法令名・番号・リンクのみで条文本文は含まないため、
  条文の中身（具体的な要件・届出義務の有無）は断定せず、e-Govのリンクで原文確認を促す。
- 「質問時に添付された資料」が提供された場合、その内容も根拠にして回答してよい。
  ただし添付資料は質問への回答にのみ使われ、案件の設備情報・調査結果には反映されない。
  資料の内容を調査・レポートに反映したい場合は、ヒアリングでの補足・
  「調査前に追記する」「修正・追加調査を依頼する」ボタンから依頼するよう案内する。
- 提供情報にも一般知識にもない事柄は、正直に「この情報からはわかりません」と伝える。
- 回答は簡潔に（目安：3〜6文か短い箇条書き）。箇条書きは「・」を使う（Markdown記法は太字 ** のみ使用可）。
- 最後の判断は所管の行政窓口や専門家への確認が必要である点を、必要に応じて一言添える。
"""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…（長いため以降省略）"


def _dump(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=1)
    except Exception:
        return str(obj)


LAW_EXTRACT_SYSTEM = """あなたは設備導入時の法令確認を支援するアシスタントです。
担当者の質問文（と設備情報）から、日本の法令データベース（e-Gov）で調べるべき
法令の名称を推定して列挙してください。

## ルール
- 質問に関係しうる日本の法令・政令・省令・規則の名称のみを挙げる（最大5件）。
- できるだけ正式名称で書く（例：「大気汚染防止法」「危険物の規制に関する政令」）。
  正式名称が不確かなときは、よく使われる通称でよい（例：「フロン排出抑制法」）。
- 質問から法令が特定できない場合は、無理に挙げず空配列にする。
- 出力は法令名の JSON 配列のみ。前置き・説明・コードブロックは書かない。
  例: ["大気汚染防止法", "消防法"]
"""


def extract_law_names(
    question: str,
    equipment_info: dict | None = None,
    config: dict | None = None,
) -> list[str]:
    """質問文（＋設備情報）から e-Gov で検索すべき法令名を推定して返す（最大5件）。
    特定できない・失敗時は空リスト。状態は変更しない。"""
    ctx = ""
    if equipment_info:
        ctx = "\n\n## 設備情報（参考）\n" + _clip(_dump(equipment_info), 2000)
    try:
        response = _llm().invoke(
            [
                SystemMessage(LAW_EXTRACT_SYSTEM),
                HumanMessage(f"# 担当者の質問\n{question}{ctx}"),
            ],
            config=config or {},
        )
        text = str(response.content).strip()
        text = re.sub(r"^```[a-zA-Z]*\n|```$", "", text).strip()
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        names = json.loads(m.group(0))
        # 重複除去（順序保持）＋空要素除去
        out: list[str] = []
        for n in names:
            s = str(n).strip()
            if s and s not in out:
                out.append(s)
        return out[:5]
    except Exception:
        logger.exception("法令名の抽出に失敗")
        return []


def build_context(
    equipment_info: dict | None,
    policy: dict | None,
    results: dict | None,
    phase_label: str,
    *,
    search_sources: list | None = None,
    attached_docs: list | None = None,
    web_results: list | None = None,
    internal_results: list | None = None,
    law_results: list | None = None,
) -> str:
    """session_state に保存済みの案件情報から、QA 用のコンテキスト文字列を組み立てる。

    policy: policy_review の interrupt データ（analysis_summary / issues / search_plan 等）
    results: results_review の interrupt データ（law_items / summary / excluded_laws 等）
    いずれも未到達のフェーズでは None のままでよい。
    """
    parts = [f"## 現在の画面\n{phase_label}"]

    if equipment_info:
        parts.append("## 設備情報（ヒアリング結果）\n" + _clip(_dump(equipment_info), _MAX_SECTION_CHARS))

    if policy:
        policy_view = {
            "分析概要":       policy.get("analysis_summary", ""),
            "調査が必要な項目": policy.get("issues", []),
            "不明・未定情報":   policy.get("unknown_items", []),
            "調査方針":       policy.get("search_plan", ""),
            "検索キーワード":   policy.get("search_keywords", []),
        }
        parts.append("## 調査方針\n" + _clip(_dump(policy_view), _MAX_SECTION_CHARS))

    if results:
        results_view = {
            "総括":             results.get("summary", ""),
            "優先度別件数":       results.get("risk_count", {}),
            "法令別の対応事項":    results.get("law_items", []),
            "非該当と判断した法令": results.get("excluded_laws", []),
            "対応情報が見つからなかった論点": results.get("uncovered_issues", []),
            "論点ごとのカバー元":  results.get("issue_coverage", {}),
        }
        parts.append("## 調査結果\n" + _clip(_dump(results_view), _MAX_SECTION_CHARS))

    if attached_docs:
        # attached_docs: [(ファイル名, 抽出テキスト), ...]
        # 質問への回答にのみ使う参考資料（案件の state には保存されない）
        blocks = [
            f"### {name}\n{_clip(text, _MAX_DOC_CHARS)}"
            for name, text in attached_docs
            if str(text).strip()
        ]
        if blocks:
            parts.append(
                "## 質問時に添付された資料\n" + _clip("\n\n".join(blocks), _MAX_DOC_SECTION_CHARS)
            )

    if web_results:
        # 質問時に担当者が任意で実行したWeb検索の結果（タイトル＋概要）。
        # 回答・資料の根拠にしてよいが、出典タイトルを添えて使う
        blocks = [
            f"- {r.get('title', '')}\n  概要: {str(r.get('snippet', ''))[:300]}"
            for r in web_results[:8]
            if r.get("title") and r.get("source") not in ("error", "unavailable")
        ]
        if blocks:
            parts.append(
                "## 質問時のWeb検索結果（参考情報）\n" + _clip("\n".join(blocks), _MAX_SECTION_CHARS)
            )

    if internal_results:
        # 質問時に担当者が任意で実行した社内文書検索の結果（タイトル＋抜粋）。
        # 回答・資料の根拠にしてよいが、出典（文書名）を添えて使う
        blocks = [
            f"- {r.get('title', '')}\n  抜粋: {str(r.get('snippet', ''))[:300]}"
            for r in internal_results[:8]
            if r.get("title") and r.get("source") not in ("error", "unavailable")
        ]
        if blocks:
            parts.append(
                "## 質問時の社内文書検索結果（参考情報）\n" + _clip("\n".join(blocks), _MAX_SECTION_CHARS)
            )

    if law_results:
        # 質問時に担当者が任意で実行した e-Gov 法令検索の結果。
        # 法令名・番号・リンクのみで条文本文は含まないため、内容の断定には使わない
        seen: set = set()
        lines = []
        for r in law_results[:8]:
            lid = r.get("law_id", "")
            if lid and lid in seen:
                continue
            if lid:
                seen.add(lid)
            if not r.get("title"):
                continue
            num = f"（{r['law_number']}）" if r.get("law_number") else ""
            url = f" {r['url']}" if r.get("url") else ""
            lines.append(f"- {r['title']}{num}{url}")
        if lines:
            parts.append(
                "## 質問時の法令検索結果（e-Gov・参考情報）\n"
                "※法令名・番号・リンクのみ（条文本文は含みません）。関連しそうな法令の"
                "存在確認・出典リンク提示に使い、条文内容は断定しないこと。\n"
                + _clip("\n".join(lines), _MAX_SECTION_CHARS)
            )

    if search_sources:
        # 「- タイトル (source)」形式で列挙する。Gemini Grounding 由来の行は
        # observability.py のマスクパターンと一致し、Langfuse 送信時に自動で
        # [REDACTED] 化される（規約対応）。LLM への入力自体はマスクされない。
        lines = [
            f"- {s.get('title', '')} ({s.get('source', '')})"
            for s in search_sources[:80]
            if s.get("title")
        ]
        parts.append(
            "## 調査で参照した情報源（タイトル一覧）\n" + _clip("\n".join(lines), _MAX_SECTION_CHARS)
        )

    return "\n\n".join(parts)


def build_history(display_messages: list, max_items: int = 6) -> str:
    """直近の会話（AI・担当者・QA）を抜粋する。「それって何？」等の指示語に追従するため。"""
    role_labels = {"ai": "AI", "user": "担当者", "qa": "QA回答"}
    lines = []
    for m in display_messages:
        label = role_labels.get(m.get("role", ""))
        if label and m.get("content"):
            lines.append(f"[{label}] {_clip(str(m['content']), 600)}")
    return _clip("\n".join(lines[-max_items:]), _MAX_HISTORY_CHARS)


def answer_question(
    question: str,
    context: str,
    history: str = "",
    config: dict | None = None,
) -> str:
    """質問1件に回答する。状態は変更しない。"""
    history_part = f"\n\n## 直近のやり取り\n{history}" if history else ""
    response = _llm().invoke(
        [
            SystemMessage(QA_SYSTEM),
            HumanMessage(
                f"# 案件情報\n\n{context}{history_part}\n\n"
                f"# 担当者からの質問\n{question}"
            ),
        ],
        config=config or {},
    )
    return str(response.content).strip()


DOC_SYSTEM = """あなたは「設備導入時 法令・届出施設確認サポートAI」の資料作成アシスタントです。
担当者の依頼に基づき、社内で使う資料（説明資料・上申資料・確認依頼・議事メモ・チェックリスト等）の
本文を作成してください。提供された案件情報・調査結果・添付資料を根拠にできます。

## 出力形式（厳守）
- Markdown のみを出力する。前置き・後書き・コードブロック記号（```）は書かない。
- 1行目は必ず「# タイトル」（資料の題名。依頼内容がわかる簡潔なもの）。
- 以降は「## 見出し」で章立てし、本文は段落・箇条書き（-）・番号リスト（1.）・表（|区切り）を使う。
- 装飾は太字（**）のみ使用可。リンク・画像・脚注は使わない。

## 内容のルール
- 法令に詳しくない読み手にもわかる言葉で書く。専門用語には短い説明を添える。
- この案件の設備情報・調査結果に即した具体的な内容にする。提供情報にないことは
  推測で断定せず、「要確認」と明示する。
- あなたは調査を行わない。調査結果の内容（対象法令・優先度・対応事項）を変更・追加しない。
- 「質問時のWeb検索結果」が提供された場合、資料の根拠に使ってよい。
  使った箇所には出典（検索結果のタイトル）を本文中に添える。
- 法令の適用有無・届出要否を断定する表現は避け、「〜の可能性」「〜の確認が必要」とする。
- 依頼が資料作成として成立しない場合（単なる質問等）でも、依頼内容を整理した簡潔なメモ資料として出力する。
"""


DOC_SLIDES_EXTRA = """
## PowerPoint向けの追加ルール（この資料はスライドに変換される）
- 「## 見出し」1つが1枚のスライドになる。章立てはスライド構成として設計する。
- 各スライドの本文は箇条書き中心で6行以内、1行は50字以内を目安に簡潔に書く。
- 長い説明は複数のスライド（見出し）に分ける。表は小さめ（5行×4列以内）にする。
"""


def generate_document(
    request: str,
    context: str,
    history: str = "",
    config: dict | None = None,
    slides: bool = False,
) -> tuple:
    """資料本文を生成し (タイトル, Markdown本文) を返す。状態は変更しない。
    slides=True の場合は PowerPoint 向け（1見出し=1スライド）の構成で生成する。"""
    history_part = f"\n\n## 直近のやり取り\n{history}" if history else ""
    response = _llm().invoke(
        [
            SystemMessage(DOC_SYSTEM + (DOC_SLIDES_EXTRA if slides else "")),
            HumanMessage(
                f"# 案件情報\n\n{context}{history_part}\n\n"
                f"# 資料作成の依頼\n{request}"
            ),
        ],
        config=config or {},
    )
    md = str(response.content).strip()
    # 出力形式が守られなかった場合に備えてコードフェンスを剥がす
    md = re.sub(r"^```[a-zA-Z]*\n|```$", "", md).strip()
    title = "資料"
    m = re.match(r"^#\s*(.+)", md)
    if m:
        title = m.group(1).strip()
        md = md[m.end():].lstrip("\n")
    return title, md
