# 設備導入時 法令・手続き確認支援AI

設備導入時に必要な法令確認・届出確認・社内手続き確認を、AIとの対話でEnd to Endで完結させるシステム。

---

## 概要

研究所に設備を導入する際、確認すべき法令・規制・届出は多岐にわたり、担当者の知識と経験に依存しがちです。本システムは、AI（GPT）が担当者から設備情報をヒアリングし、e-Gov API・Gemini Web検索・社内文書RAGを組み合わせて調査を自動実行、対応が必要な事項と確認先をレポートにまとめます。

### 主な特徴

- **資料アップロードで手入力を削減**：仕様書・設備リスト（PDF/Word/Excel/PowerPoint/テキスト）から設備情報を構造化抽出し、担当者は確認・修正するだけ。未記入の項目のみAIが1問ずつ質問します。
- **Agentic Search**：固定キーワード検索ではなく、AIが検索結果（e-Gov法令API・Gemini Web検索・社内文書）を見ながら「次に何を調べるか」を自律的に判断して収集します。横浜市・神奈川県の条例、省庁ガイドラインのWeb検索も必ず実施します。
- **社内文書の Agentic RAG**：サイドバーから社内規定・過去の届出事例を登録すると、Agentic Search 配下のRAGサブエージェント（クエリ展開→ベクトル検索→関連性評価→再検索）が社内文書も調査します（Chroma / FAISS 切替可）。
- **条文原文にもとづく判定**：結果統合の前に主要候補法令の条文（適用範囲・届出義務・数量閾値）をe-Gov原文から取得してLLMに渡し、冷媒充填量などの**数量を条文の閾値と照合**して判定します。条番号もe-Gov原文と照合して確定（ハルシネーション防止）。
- **Human-in-the-loop（やり直し対応）**：方針確認・結果確認・レポート確認の3か所で担当者が介入できます。担当者の追記・修正依頼・再調査依頼は実際に調査・統合・レポートに反映され、**結果確認／レポート確認から調査フェーズへ戻って何度でも再調査**できます。
- **ケースメモリ（使うほど賢くなる）**：レポート承認済みの案件を事例として保存し、次回の類似案件の分析時に自動参照します（Case-Based Reasoning）。
- **漏れ防止の工夫**：ルールベース必須法令の決定的シード検索、主要11法令領域の「該当/非該当（理由つき）」強制振り分け、論点ごとの網羅性検証（カバー元の法令名つき）、不明・未定情報の明示。レポートには「法令確認の全体像」セクションを常設し、ルールベース必須法令×最終法令リストの突合結果と、該当・要確認・非該当の全件一覧を表示します。
- **初心者にも読めるレポート**：総括とバッジの読み方、全届出・社内対応を期限順にまとめた**対応チェックリスト（時系列）**、条文の折りたたみ表示。
- **AI利用コストの見える化**：サイドバーにLLM・Web検索のトークン数と概算コスト（$/円）を案件単位で表示します。
- **Langfuse によるトレース・コスト計測（任意）**：`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` を設定すると、全LLM呼び出し（LangGraph 各ノード）と Gemini Web検索のリクエスト内容・トークン数・コストが Langfuse に記録されます。トレースは案件（thread_id）単位でセッションとしてグルーピングされます。

---

## 処理フロー

```
0. 資料アップロード ★  仕様書等から設備情報を抽出し、担当者が確認・確定（任意）
        ↓
1. ヒアリング        未記入の項目だけをAIが1問ずつ質問（質問の目的・回答例つき）
        ↓
2. 情報整理・分析     論点・不明情報・調査方針を生成
                    （🧠 ケースメモリから類似の過去案件を自動参照）
        ↓
3. 方針確認 ★        担当者が調査方針を確認・承認
                    （「調査前に追記」で指示を追記すると調査に反映される）
        ↓
4. 自動調査          e-Gov API（国法令）＋ Gemini Web検索 ＋ 社内文書RAG を
                    AIが自律的に判断しながら収集（Agentic Search）
                    → 論点ごとの網羅性検証も実施
        ↓
5. 結果統合          主要候補法令の条文をe-Gov原文から取得し、
                    数量閾値と照合して法令別の対応事項を整理（条番号も原文照合）
        ↓
6. 結果確認 ★        担当者が法令別の対応事項・条文・届出先を確認
                    └→「不足あり・追加調査」で 4. に戻って再調査（何度でも可）
        ↓
7. レポート生成・確認 ★ 担当者がレポートを確認し、いずれかを選択：
                    ・承認 → 完了（🧠 ケースメモリに事例を自動保存）
                    ・文面修正 → AIが表現を直して再生成
                    ・追加調査 → 4. に戻って再調査
        ↓
8. 完了              HTMLレポートをダウンロード（outputs/ に自動保存）
```

★ = Human-in-the-loop ポイント（LangGraph `interrupt()`）。
担当者の追記・修正依頼・再調査依頼はすべて `interrupt()` の戻り値として
ワークフローに反映される。6.・7. からは調査フェーズへ戻る**やり直しループ**を備える。
詳細なフロー図は `flow_diagram.html` を参照。

---

## システム構成

```
法令検索AI_claude_code/
├── app.py                      Streamlit UI（メインアプリ・コスト表示）
├── backend/
│   ├── state.py                LangGraph 状態定義（TypedDict）
│   ├── fields.py               ヒアリング11項目の共通定義
│   ├── prompts.py              日本語プロンプト集
│   ├── workflow.py             LangGraph ワークフロー（7ノード＋やり直しループ）
│   ├── doc_intake.py           アップロード資料からの設備情報抽出
│   ├── rag_agent.py            社内文書の Agentic RAG サブエージェント
│   ├── case_memory.py          ケースメモリ（承認済み案件の保存・想起）
│   ├── report_gen.py           HTML レポート生成（チェックリスト・条文つき）
│   └── tools/
│       ├── egov.py             e-Gov API ラッパー（法令検索・条文取得・条番号照合）
│       ├── web_search.py       Gemini Google Search Grounding（Web検索）
│       └── internal_docs.py    社内文書ベクトルストア（Chroma/FAISS・登録/削除/検索）
├── internal_docs_index/        社内文書・ケースメモリの永続化先（Git管理外）
├── outputs/                    生成レポート保存先（承認時に自動保存）
├── flow_diagram.html           処理フロー図
├── requirements.txt
├── .env.example                環境変数テンプレート
└── README.md
```

---

## AI構成

| 用途 | PoC環境 | 本番環境 |
|------|---------|---------|
| 主要LLM（ヒアリング・分析・統合） | OpenAI API（GPT-4o等） | Azure OpenAI（GPT系） |
| Web検索AI | Gemini Google Search Grounding | Gemini Google Search Grounding |
| 法令検索 | e-Gov API（共通） | e-Gov API（共通） |
| 埋め込み（社内文書・ケースメモリ） | OpenAI（text-embedding-3-small） | Azure OpenAI |
| ベクトルストア | Chroma（既定）/ FAISS | Chroma（既定）/ FAISS |

---

## セットアップ

### 前提条件

- Python 3.11 以上
- OpenAI API キー（PoC環境）または Azure OpenAI リソース（本番環境）
- Gemini API キー（Web検索、任意）

### 1. 仮想環境の作成とパッケージのインストール

```bash
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### 2. 環境変数の設定

```bash
copy .env.example .env
```

`.env` を編集して必要な値を設定します。

**PoC環境（OpenAI）の最小設定：**

```env
LLM_MODE=poc
POC_LLM_API_KEY=sk-...        # OpenAI APIキー
POC_LLM_MODEL=gpt-4o
```

**本番環境（Azure OpenAI）に切り替える場合：**

```env
LLM_MODE=prod
PROD_LLM_API_KEY=...
PROD_LLM_ENDPOINT=https://<リソース名>.openai.azure.com/
PROD_LLM_API_VERSION=2024-02-01
PROD_LLM_DEPLOYMENT=<デプロイ名>
```

**Web検索を有効にする場合（任意）：**

```env
GEMINI_API_KEY=...
GEMINI_WEB_SEARCH_MODEL=gemini-2.0-flash
```

### 3. アプリの起動

```bash
.venv\Scripts\streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。

---

## 環境変数一覧

| 変数名 | 説明 | 必須 |
|--------|------|------|
| `LLM_MODE` | `poc` または `prod` | ✅ |
| `POC_LLM_API_KEY` | OpenAI APIキー（PoC用） | PoC時 |
| `POC_LLM_MODEL` | OpenAIモデルID（例：`gpt-4o`） | PoC時 |
| `PROD_LLM_API_KEY` | Azure OpenAI APIキー（本番用） | 本番時 |
| `PROD_LLM_ENDPOINT` | Azure OpenAI エンドポイントURL | 本番時 |
| `PROD_LLM_API_VERSION` | Azure OpenAI APIバージョン | 本番時 |
| `PROD_LLM_DEPLOYMENT` | Azure OpenAI デプロイ名 | 本番時 |
| `GEMINI_API_KEY` | Gemini APIキー（Web検索用） | 任意 |
| `GEMINI_WEB_SEARCH_MODEL` | GeminiモデルID | 任意 |
| `POC_EMBEDDING_MODEL` | 埋め込みモデル（既定 `text-embedding-3-small`。変更時は社内文書の再登録が必要） | 任意 |
| `POC_EMBEDDING_API_KEY` | 埋め込み用APIキー（未設定時は `POC_LLM_API_KEY` を流用） | 任意 |
| `PROD_EMBEDDING_DEPLOYMENT` | Azure 埋め込みデプロイ名（未設定の項目は `PROD_LLM_*` を流用） | 本番でRAG利用時 |
| `VECTOR_STORE` | ベクトルストア `chroma`（既定）/ `faiss` | 任意 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse のAPIキー。設定すると全LLM呼び出し・Web検索のトレースとコストが Langfuse に記録される | 任意 |
| `LANGFUSE_HOST` | Langfuse のURL（既定 `https://cloud.langfuse.com`。セルフホスト時に変更） | 任意 |
| `LLM_COST_INPUT_PER_1M` ほか | サイドバー概算コスト表示の単価設定（詳細は `.env.example` 参照） | 任意 |

---

## ヒアリング項目

AIが対話形式で1問ずつ収集する11項目（不明・未定の回答も受け付ける）：

| 項目 | 内容 |
|------|------|
| 設備種別 | 実験装置、加工装置、評価設備 等 |
| 設置場所 | 建屋・階・部屋番号 |
| 用途・目的 | 設備の使用目的 |
| 稼働予定日 | 稼働開始予定日 |
| 薬品・ガス | 有機溶剤、特定化学物質、高圧ガス等の使用有無 |
| 火気・排気 | 火気・熱源・排気の発生有無 |
| 排水 | 排水の発生有無・種類 |
| 騒音・振動 | 騒音・振動の発生有無 |
| 放射線・X線 | 放射線・X線装置への該当有無 |
| 建屋改修 | 電気工事・配管工事・建屋改修の有無 |
| その他情報 | 仕様・設置環境・搬入経路・メーカー名など上記以外の補足 |

> 薬品・ガス・火気・排水・放射線の項目は、「使用する可能性があるのに『なし』と回答すると法令確認が漏れる」ため、不確かな場合は『不明』での回答を案内します。

---

## 調査対象となる主な法令

- 労働安全衛生法・有機溶剤中毒予防規則・特定化学物質障害予防規則
- 消防法・危険物の規制に関する政令
- 大気汚染防止法・水質汚濁防止法
- 高圧ガス保安法
- 建築基準法
- 放射線障害防止法
- 横浜市条例・神奈川県条例
- 社内設備安全審査規程

---

## 技術スタック

| 技術 | 用途 |
|------|------|
| [Streamlit](https://streamlit.io/) | Web UI |
| [LangGraph](https://github.com/langchain-ai/langgraph) | ワークフロー・状態管理・Human-in-the-loop |
| [LangChain](https://github.com/langchain-ai/langchain) | LLM抽象化・ツール呼び出し |
| OpenAI / Azure OpenAI | 主要LLM（GPT） |
| Gemini Google Search Grounding | 公開Web法令情報の検索 |
| [e-Gov API](https://laws.e-gov.go.jp/docs/api/) | 国法令の検索・条文取得・条番号照合 |
| Chroma / FAISS | 社内文書・ケースメモリのベクトルDB |

---

## 注意事項

- 本システムの出力は**参考情報**です。最終的な法令判断・届出要否の確定は担当者・関係部署・所轄機関が行ってください。
- 社内文書の内容は Gemini API 等の外部AIに送信しません。
- APIキーはソースコードやログに含めず、`.env` または Secrets 管理ツールで管理してください。
- `.env` は `.gitignore` に追加してリポジトリにコミットしないでください。
