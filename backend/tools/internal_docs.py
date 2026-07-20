"""社内文書のハイブリッド検索（Agentic RAG 用）

事前に登録した社内文書（社内規定・基準・過去の届出事例・手続きマニュアル等）を
チャンク分割・埋め込みして永続化し、調査フェーズの Agentic 検索ループから
e-Gov API・Web検索と並ぶ検索タイプ（search_type="internal"）として利用する。

- ベクトルストア: VECTOR_STORE 環境変数で切替（chroma=既定 / faiss）
- 埋め込み: LLM_MODE 環境変数で切替（poc=OpenAI / prod=Azure OpenAI）
- 検索: ベクトル検索（意味的な類似）と BM25（キーワード一致）を
  Reciprocal Rank Fusion で統合するハイブリッド検索。型番・法令名など
  ベクトル検索だけでは表記ゆれ扱いで取りこぼしやすい完全一致語を補う。
  BM25 は日本語の分かち書きがないため文字bi-gramで代替トークナイズする。
- 登録済みファイルの一覧・重複判定・削除のため registry.json を併置する
"""

import os
import re
import json
import hashlib
import datetime
import logging
import threading
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

logger = logging.getLogger(__name__)

# インデックスの永続化先（プロジェクト直下）
INDEX_DIR = Path(__file__).resolve().parents[2] / "internal_docs_index"
REGISTRY_PATH = INDEX_DIR / "registry.json"

# 1ファイルあたりの登録テキスト上限（埋め込みコストの暴走防止。超過分は切り捨て）
MAX_DOC_CHARS = 50000

# チャンク分割設定（日本語文書向けの区切り優先順）
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", "。", "、", " ", ""],
)

_LOCK = threading.Lock()
_STORE = None  # プロセス内キャッシュ
_BM25 = None   # プロセス内キャッシュ（登録・削除のたび全文書から再構築するため無効化が必要）


# ─── 埋め込みファクトリ（LLM_MODE で OpenAI / Azure を切替）──────
def _embeddings():
    """EMBEDDING 系の環境変数を優先し、未設定なら LLM 系の設定にフォールバックする
    （.env.example の定義と一致させる。埋め込みが LLM と別リソースの構成に対応）。
    ※ 埋め込みモデルを変更した場合、既存インデックスと次元が合わなくなるため
    　 社内文書・ケースメモリの再登録が必要。"""
    mode = os.getenv("LLM_MODE", "poc").lower()
    if mode == "prod":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("PROD_EMBEDDING_DEPLOYMENT", ""),
            azure_endpoint=(
                os.getenv("PROD_EMBEDDING_ENDPOINT", "")
                or os.getenv("PROD_LLM_ENDPOINT", "")
            ),
            api_key=(
                os.getenv("PROD_EMBEDDING_API_KEY", "")
                or os.getenv("PROD_LLM_API_KEY", "")
            ),
            openai_api_version=(
                os.getenv("PROD_EMBEDDING_API_VERSION", "")
                or os.getenv("PROD_LLM_API_VERSION", "2024-02-01")
            ),
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=os.getenv("POC_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=(
            os.getenv("POC_EMBEDDING_API_KEY")
            or os.getenv("POC_LLM_API_KEY")
        ),
    )


def _backend() -> str:
    return os.getenv("VECTOR_STORE", "chroma").lower()


# ─── ベクトルストア ───────────────────────────────────────────────
def _faiss_dir() -> Path:
    return INDEX_DIR / "faiss"


def _load_store():
    """既存インデックスをロードする。未作成なら None（faiss）または空ストア（chroma）。"""
    global _STORE
    if _STORE is not None:
        return _STORE
    if _backend() == "faiss":
        from langchain_community.vectorstores import FAISS
        if (_faiss_dir() / "index.faiss").exists():
            # 自前で保存したローカルインデックスのため deserialization を許可する
            _STORE = FAISS.load_local(
                str(_faiss_dir()), _embeddings(),
                allow_dangerous_deserialization=True,
            )
    else:
        from langchain_chroma import Chroma
        _STORE = Chroma(
            collection_name="internal_docs",
            embedding_function=_embeddings(),
            persist_directory=str(INDEX_DIR / "chroma"),
        )
    return _STORE


def _persist_if_faiss() -> None:
    if _backend() == "faiss" and _STORE is not None:
        _faiss_dir().mkdir(parents=True, exist_ok=True)
        _STORE.save_local(str(_faiss_dir()))


# ─── BM25（キーワード検索）───────────────────────────────────────
def _bm25_tokenize(text: str) -> list[str]:
    """BM25用の簡易トークナイズ。日本語は分かち書きがなく単純な空白split
    では文全体が1トークンになってしまうため、文字bi-gramで代替する
    （形態素解析器を追加せずに部分一致・型番等の完全一致を拾える、
    CJK全文検索で広く使われる手法）。"""
    t = re.sub(r"\s+", "", text)
    if len(t) < 2:
        return [t] if t else []
    return [t[i:i + 2] for i in range(len(t) - 1)]


def _all_documents() -> list[Document]:
    """インデックス済みの全チャンクを Document のリストで返す（BM25構築用）。"""
    store = _load_store()
    if store is None:
        return []
    if _backend() == "faiss":
        return list(store.docstore._dict.values())
    result = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(result.get("documents", []) or [], result.get("metadatas", []) or [])
    ]


def _load_bm25():
    """BM25Retriever をプロセス内キャッシュから返す（未構築・無効化後は再構築）。
    未登録時は None。"""
    global _BM25
    if _BM25 is not None:
        return _BM25
    docs = _all_documents()
    if not docs:
        return None
    _BM25 = BM25Retriever.from_documents(docs, preprocess_func=_bm25_tokenize)
    return _BM25


def _invalidate_bm25() -> None:
    global _BM25
    _BM25 = None


def _hybrid_search(query: str, k: int) -> list[Document]:
    """ベクトル検索とBM25（キーワード）検索を Reciprocal Rank Fusion で統合する
    ハイブリッド検索。表記ゆれに強いベクトル検索と、型番・法令名などの
    完全一致に強いBM25を組み合わせ、どちらか一方だけでは拾えない結果を補う。
    BM25構築に失敗・未登録の場合はベクトル検索のみにフォールバックする。"""
    store = _load_store()
    if store is None:
        return []
    vector_retriever = store.as_retriever(search_kwargs={"k": k})
    try:
        bm25 = _load_bm25()
    except Exception:
        logger.exception("BM25インデックスの構築に失敗（ベクトル検索のみで継続）")
        bm25 = None
    if bm25 is None:
        return vector_retriever.invoke(query)
    bm25.k = k
    ensemble = EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[0.5, 0.5])
    return ensemble.invoke(query)[:k]


# ─── 登録ファイル台帳 ─────────────────────────────────────────────
def _load_registry() -> list[dict]:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_registry(reg: list[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_registered() -> list[dict]:
    """登録済みファイルの一覧（doc_id / name / chunks / registered_at）。

    doc_id は内容ハッシュ（sha1）で、削除・特定の一意キー。同じファイル名で
    内容の異なる資料は別 doc_id の別エントリとして登録される。
    """
    return [
        {"doc_id": r.get("sha1", ""), "name": r["name"],
         "chunks": r.get("chunks", 0),
         "registered_at": r.get("registered_at", "")}
        for r in _load_registry()
    ]


def internal_docs_available() -> bool:
    """社内文書が1件でも登録されているか（検索タイプの有効判定に使う）。"""
    return bool(_load_registry())


# ─── 登録・削除 ───────────────────────────────────────────────────
def ingest_files(files: list[tuple[str, bytes]]) -> dict:
    """[(ファイル名, バイト列), ...] を抽出→分割→埋め込みして登録する。
    返り値: {"added": [名前], "skipped": [名前(登録済み)],
             "failed": [名前(抽出不可)], "truncated": [名前(上限で切り捨て)]}
    """
    # 循環importを避けるため遅延import（doc_intake → workflow → tools）
    from backend.doc_intake import extract_text_from_file

    global _STORE
    added, skipped, failed, truncated = [], [], [], []

    with _LOCK:
        reg = _load_registry()
        known_hashes = {r["sha1"] for r in reg}

        for name, data in files:
            sha1 = hashlib.sha1(data).hexdigest()
            if sha1 in known_hashes:
                skipped.append(name)
                continue
            text = extract_text_from_file(name, data)
            if not text.strip():
                failed.append(name)
                continue
            if len(text) > MAX_DOC_CHARS:
                text = text[:MAX_DOC_CHARS]
                truncated.append(name)

            chunks = _SPLITTER.split_text(text)
            ids = [f"{sha1}-{i}" for i in range(len(chunks))]
            docs = [
                Document(
                    page_content=c,
                    metadata={"source": name, "chunk_no": i + 1},
                )
                for i, c in enumerate(chunks)
            ]

            store = _load_store()
            if store is None:
                # faiss は空インデックスを作れないため、初回登録時に生成する
                from langchain_community.vectorstores import FAISS
                _STORE = FAISS.from_documents(docs, _embeddings(), ids=ids)
            else:
                store.add_documents(docs, ids=ids)
            _persist_if_faiss()

            reg.append({
                "name": name,
                "sha1": sha1,
                "chunk_ids": ids,
                "chunks": len(chunks),
                "registered_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            known_hashes.add(sha1)
            added.append(name)

        if added:
            _save_registry(reg)
            _invalidate_bm25()

    return {"added": added, "skipped": skipped, "failed": failed, "truncated": truncated}


def delete_document(doc_id: str) -> bool:
    """登録済み文書を doc_id（内容ハッシュ）指定で削除する。

    ファイル名ではなく doc_id をキーにする。同じファイル名で内容の異なる
    資料が複数登録されている場合、名前で削除すると1件分のベクトルしか
    消せないまま台帳から同名エントリを全消しし、検索にヒットし続ける
    孤児ベクトルが残るため。

    ベクトルの削除に失敗した場合は台帳を変更せず False を返す
    （台帳だけ消して孤児ベクトルを残さない）。
    """
    global _STORE
    with _LOCK:
        reg = _load_registry()
        entry = next((r for r in reg if r.get("sha1") == doc_id), None)
        if entry is None:
            return False
        store = _load_store()
        if store is not None and entry.get("chunk_ids"):
            try:
                store.delete(ids=entry["chunk_ids"])
                _persist_if_faiss()
            except Exception:
                logger.exception(
                    "社内文書インデックスからの削除に失敗: %s (%s)",
                    entry.get("name"), doc_id,
                )
                return False
        _save_registry([r for r in reg if r.get("sha1") != doc_id])
        _invalidate_bm25()
    return True


def delete_all() -> int:
    """登録済みの全ファイルをインデックスと台帳から削除する。返り値は削除ファイル数。"""
    global _STORE
    with _LOCK:
        reg = _load_registry()
        if not reg:
            return 0
        store = _load_store()
        all_ids = [i for r in reg for i in r.get("chunk_ids", [])]
        if store is not None and all_ids:
            try:
                store.delete(ids=all_ids)
                _persist_if_faiss()
            except Exception:
                # 台帳だけ消すと孤児ベクトルが検索に残るため、台帳は変更しない
                logger.exception("社内文書インデックスの全削除に失敗")
                return 0
        _save_registry([])
        _invalidate_bm25()
    return len(reg)


# ─── 検索 ─────────────────────────────────────────────────────────
def retrieve_chunks(query: str, k: int = 4) -> list[dict]:
    """Agentic RAG サブエージェント用の低レベル検索。
    チャンク全文とID付きで返す（関連性評価に全文が必要なため）。
    未登録・失敗時は空リスト。
    """
    if not internal_docs_available():
        return []
    try:
        hits = _hybrid_search(query, k)
        return [
            {
                "id": f"{d.metadata.get('source', '?')}#{d.metadata.get('chunk_no', '?')}",
                "source_file": d.metadata.get("source", "?"),
                "chunk_no": d.metadata.get("chunk_no", 0),
                "content": d.page_content,
            }
            for d in hits
        ]
    except Exception:
        logger.exception("社内文書チャンク検索に失敗: %s", query)
        return []


def search_internal_docs(query: str, k: int = 5) -> list[dict]:
    """社内文書をハイブリッド検索（ベクトル＋BM25）し、search_web と同形式の
    結果リストを返す。未登録・インデックス無しは空リスト。失敗時は
    source="error" の1件を返す（呼び出し側の _process_web_results がエラー表示に変換する）。
    """
    if not internal_docs_available():
        return []
    try:
        hits = _hybrid_search(query, k)
        return [
            {
                "title": (
                    f"社内文書: {d.metadata.get('source', '?')}"
                    f"（抜粋{d.metadata.get('chunk_no', '?')}）"
                ),
                "snippet": d.page_content[:300],
                "url": "",
                "source": "社内文書",
            }
            for d in hits
        ]
    except Exception as e:
        logger.exception("社内文書検索に失敗: %s", query)
        return [{"title": f"社内文書検索エラー: {e}", "snippet": "", "url": "", "source": "error"}]
