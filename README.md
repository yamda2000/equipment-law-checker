# 設備導入時 法令・手続き確認支援AI

設備導入時に必要な法令確認・届出確認・社内手続き確認を、AIとの対話でEnd to Endで完結させるシステム。

---

## 概要

研究所に設備を導入する際、確認すべき法令・規制・届出は多岐にわたり、担当者の知識と経験に依存しがちです。本システムは、AI（GPT）が担当者から設備情報をヒアリングし、e-Gov API・Gemini Web検索・社内文書RAGを組み合わせて調査を自動実行、対応が必要な事項と確認先をレポートにまとめます。

---

## 処理フロー

```
1. ヒアリング        AIが設備情報を10項目ヒアリング（対話形式）
        ↓
2. 論点整理          収集情報から確認が必要な論点・調査方針を生成
        ↓
3. 方針確認 ★        担当者が調査方針を確認・承認（Human-in-the-loop）
        ↓
4. 自動調査          e-Gov API（国法令）+ Gemini Web検索 を並行実行
        ↓
5. 結果確認 ★        担当者が対応事項リストを1件ずつ採用/修正/不要を判断
        ↓
6. レポート生成       HTMLレポートを自動作成
        ↓
7. 生成物確認 ★       担当者がレポートを確認・承認
        ↓
8. 完了              HTMLレポートをダウンロード
```

★ = Human-in-the-loop ポイント（LangGraph `interrupt()`）

---

## システム構成

```
法令検索AI_claude_code/
├── app.py                      Streamlit UI（メインアプリ）
├── prototype/
│   └── app.py                  UIプロトタイプ（参考用）
├── backend/
│   ├── state.py                LangGraph 状態定義（TypedDict）
│   ├── prompts.py              日本語プロンプト集
│   ├── workflow.py             LangGraph ワークフロー（5ノード）
│   ├── report_gen.py           HTML レポート生成
│   └── tools/
│       ├── egov.py             e-Gov API ラッパー（v2/v1フォールバック）
│       └── web_search.py       Gemini Google Search Grounding ラッパー
├── docs/                       社内文書置き場（Chroma RAG用、将来拡張）
├── outputs/                    生成レポート保存先
├── requirements.txt
├── .env.example                環境変数テンプレート
└── README.md
```

---

## AI構成

| 用途 | PoC環境 | 本番環境 |
|------|---------|---------|
| 主要LLM（ヒアリング・分析・合成） | OpenAI API（GPT-4o等） | Azure OpenAI（GPT系） |
| Web検索AI | Gemini Google Search Grounding | Gemini Google Search Grounding |
| 法令検索 | e-Gov API（共通） | e-Gov API（共通） |
| 社内文書RAG | Chroma（将来実装） | Chroma（将来実装） |

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

---

## ヒアリング項目

AIが対話形式で収集する10項目：

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
| [e-Gov API](https://laws.e-gov.go.jp/docs/api/) | 国法令テキストの検索 |
| Chroma | 社内文書ベクトルDB（将来実装） |

---

## 注意事項

- 本システムの出力は**参考情報**です。最終的な法令判断・届出要否の確定は担当者・関係部署・所轄機関が行ってください。
- 社内文書の内容は Gemini API 等の外部AIに送信しません。
- APIキーはソースコードやログに含めず、`.env` または Secrets 管理ツールで管理してください。
- `.env` は `.gitignore` に追加してリポジトリにコミットしないでください。
