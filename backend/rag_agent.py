"""社内文書の Agentic RAG サブエージェント

Agentic Search（workflow.search_node）が search_type="internal" を選んだときに
呼ばれる下位エージェント。単発のベクトル検索ではなく、

  ① 質問からの検索クエリ展開（multi-query）
  ② ベクトル検索（チャンク取得）
  ③ 取得チャンクの関連性評価と充足判断（reflection）
  ④ 不足時は追加クエリを生成して再検索

を LLM が自律的に最大 MAX_RAG_ROUNDS 回繰り返し、質問に関連する
チャンクだけを search_web と同形式の結果リストで返す。
LLM 呼び出しに失敗した場合は単発ベクトル検索にフォールバックする。
"""

import logging
from typing import Callable, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from backend.prompts import RAG_QUERY_SYSTEM, RAG_GRADE_SYSTEM
from backend.tools.internal_docs import internal_docs_available, retrieve_chunks

logger = logging.getLogger(__name__)

MAX_RAG_ROUNDS = 3        # クエリ展開→検索→評価 の最大ラウンド数
QUERIES_PER_ROUND = 3     # 1ラウンドで実行する検索クエリの上限
CHUNKS_PER_QUERY = 4      # 1クエリで取得するチャンク数


class RagQueries(BaseModel):
    queries: list[str] = Field(
        description="ベクトル検索用の短いクエリ（2〜3個。互いに観点・言い換えを変えること）",
    )


class RagAssessment(BaseModel):
    relevant_ids: list[str] = Field(
        default_factory=list,
        description="質問に関連するチャンクのID（提示された一覧のIDのまま）",
    )
    sufficient: bool = Field(
        description="True=質問に答えるのに十分な情報が集まった（または社内文書にこれ以上の情報はない）",
    )
    followup_queries: list[str] = Field(
        default_factory=list,
        description="sufficient=False の場合の追加検索クエリ（最大2個・既出クエリと観点を変える）",
    )


def agentic_internal_search(
    question: str,
    context_hint: str = "",
    on_progress: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """質問に対して社内文書を Agentic RAG で調査し、関連チャンクを返す。

    question:     外側の Agentic Search が立てた調べたいこと（論点・質問）
    context_hint: 設備種別などの補足コンテキスト
    on_progress:  進捗ログのコールバック（検索ログ・ライブ表示に流す）
    """
    if not internal_docs_available():
        return []

    def emit(msg: str) -> None:
        if on_progress:
            try:
                on_progress(msg)
            except Exception:
                pass

    from backend.workflow import _llm  # 循環importを避けるため遅延import

    hint = f"\n\n## 補足コンテキスト（設備情報）\n{context_hint}" if context_hint else ""

    # ① クエリ展開（失敗しても質問そのものをクエリにして続行する）
    try:
        query_llm = _llm(max_tokens=1000).with_structured_output(RagQueries)
        plan: RagQueries = query_llm.invoke([
            SystemMessage(RAG_QUERY_SYSTEM),
            HumanMessage(f"## 調べたいこと\n{question}{hint}"),
        ])
        queries = [q.strip() for q in plan.queries if q.strip()][:QUERIES_PER_ROUND]
    except Exception:
        logger.exception("Agentic RAG のクエリ展開に失敗: %s", question)
        queries = []
    if not queries:
        queries = [question]

    grade_llm = _llm(max_tokens=2000).with_structured_output(RagAssessment)
    collected: dict[str, dict] = {}   # id -> チャンク
    relevant_ids: set[str] = set()
    executed: set[str] = set()
    graded_ok = False

    for round_no in range(1, MAX_RAG_ROUNDS + 1):
        # ② ベクトル検索
        new_in_round = 0
        for q in queries:
            if q in executed:
                continue
            executed.add(q)
            chunks = retrieve_chunks(q, k=CHUNKS_PER_QUERY)
            new = [c for c in chunks if c["id"] not in collected]
            for c in new:
                collected[c["id"]] = c
            new_in_round += len(new)
            emit(f"　└ 📁 [社内文書検索{round_no}周目] 「{q}」→ 新規{len(new)}件")

        if not collected:
            emit("　└ 📁 [社内文書検索] 該当箇所なし")
            return []

        # 追加検索で新規チャンクが得られなければ、再評価しても結果は変わらない
        if round_no > 1 and new_in_round == 0:
            emit("　└ 📁 [社内文書検索] 新しい該当箇所なし → 前回の評価で確定")
            break

        # ③ 関連性評価・充足判断
        # （失敗しても全体を中断せず、直前ラウンドまでの評価結果で確定する）
        listing = "\n\n".join(
            f"[ID: {c['id']}]\n{c['content'][:500]}"
            for c in collected.values()
        )
        try:
            assessment: RagAssessment = grade_llm.invoke([
                SystemMessage(RAG_GRADE_SYSTEM),
                HumanMessage(
                    f"## 調べたいこと\n{question}{hint}\n\n"
                    f"## 実行済みクエリ\n" + "\n".join(f"- {q}" for q in sorted(executed))
                    + f"\n\n## 取得済みチャンク一覧\n{listing}"
                ),
            ])
        except Exception:
            logger.exception("Agentic RAG の関連性評価に失敗（%dラウンド目）: %s", round_no, question)
            emit("　└ ⚠️ [社内文書検索] 内容の評価に失敗 → これまでの評価結果で確定します")
            break
        graded_ok = True
        relevant_ids = {i for i in assessment.relevant_ids if i in collected}

        # ④ 充足していれば終了、不足なら追加クエリで再検索
        followups = [
            q.strip() for q in assessment.followup_queries
            if q.strip() and q.strip() not in executed
        ][:2]
        if assessment.sufficient or not followups or round_no == MAX_RAG_ROUNDS:
            emit(
                f"　└ 📁 [社内文書検索] 確認完了（{round_no}周・"
                f"関連する記載 {len(relevant_ids)}/{len(collected)}件）"
            )
            break
        emit("　└ 📁 [社内文書検索] 情報不足と判断 → 別の言葉で再検索")
        queries = followups

    if not graded_ok:
        # 1回も評価できなかった場合は選別せず、収集チャンクをそのまま返す
        # （展開クエリでヒットした分なので、後段の統合LLMが取捨選択できる）
        relevant_ids = set(collected.keys())

    return [
        {
            "title": f"社内文書: {c['source_file']}（抜粋{c['chunk_no']}）",
            "snippet": c["content"][:300],
            "url": "",
            "source": "社内文書",
        }
        for cid, c in collected.items() if cid in relevant_ids
    ]
