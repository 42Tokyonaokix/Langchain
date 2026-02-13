# デジタルグリッド RAGチャットボット

電力約款・システムマニュアルに基づいて回答するカスタマーサポートAIです。

## 特徴

### 検索システム
- **ハイブリッド検索**: BM25（キーワード）+ Vector（意味）の並列検索
- **RRF統合**: Reciprocal Rank Fusionで検索結果を最適化
- **電圧タイプフィルタ**: 高圧/低圧/特別高圧の自動判別・絞り込み

### インデックス作成
- **LLM構造解析**: GPT-4o-miniで文書構造（章・節）を自動認識
- **スマートチャンキング**: 第X条、(1)、イ などの法的文書構造で分割
- **リッチメタデータ**: エリア、電圧タイプ、ドキュメント種別を自動付与

### エージェント
- **ReActアーキテクチャ**: LangGraphによる思考→行動→観察ループ
- **ツール呼び出し**: 必要に応じて検索を実行し、根拠に基づいた回答

### 評価
- **ローカル評価**: LLMスコアリングでCSV出力
- **LangSmith連携**: 評価・モニタリング機能

## セットアップ

### 1. 環境構築

```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージインストール
pip install -r requirements.txt

# MeCab（日本語トークナイザ）
sudo apt install mecab libmecab-dev mecab-ipadic-utf8
```

### 2. 環境変数設定

`.env`ファイルを作成:

```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

### 3. ドキュメント配置

`data/documents/`フォルダに検索対象のドキュメントを配置:

```
data/documents/
├── 約款等/
│   ├── 電気需給約款.pdf
│   └── ...
├── 配送約款等/
│   └── 託送供給等約款_*.pdf
├── system_manual/
│   └── システムマニュアル.pdf
└── qa_list.csv
```

### 4. インデックス作成

```bash
python scripts/run_index.py --force
```

※ 初回のみ実行。ドキュメント更新時は`--force`で再作成。

## 使い方

### チャットボット起動

```bash
python scripts/run_chatbot.py
```

### 評価実行

```bash
# ローカル評価（全件）
python scripts/run_evaluate_local.py

# 件数制限（テスト用）
python scripts/run_evaluate_local.py --max=10

# LangSmith評価
python scripts/run_create_dataset.py  # データセット作成
python scripts/run_evaluate.py        # 評価実行
```

## ファイル構成

```
.
├── scripts/                        # 実行スクリプト
│   ├── run_chatbot.py              # チャットボット起動
│   ├── run_index.py                # インデックス作成（基本）
│   ├── run_index_llamaparse.py     # インデックス作成（LlamaParse）
│   ├── run_evaluate_local.py       # ローカル評価
│   ├── run_evaluate.py             # LangSmith評価
│   └── run_create_dataset.py       # データセット作成
│
├── src/
│   ├── chatbot/
│   │   └── agent.py                # ReActエージェント
│   │
│   ├── rag/                        # 検索システム
│   │   ├── search.py               # ベクトル検索
│   │   ├── hybrid.py               # ハイブリッド検索（基本）
│   │   ├── hybrid_llamaparse.py    # ハイブリッド検索（LlamaParse）
│   │   ├── llamaindex_hybrid.py    # LlamaIndex版ハイブリッド検索
│   │   └── tokenizer.py            # 日本語トークナイザ
│   │
│   ├── indexing/                   # インデックス作成
│   │   ├── basic.py                # 基本インデックス
│   │   ├── agentic.py              # LLM構造解析インデックス
│   │   ├── extractors.py           # テキスト抽出
│   │   ├── llamaindex_pipeline.py  # LlamaIndexパイプライン
│   │   └── llamaparse_agentic.py   # LlamaParse+LLM解析
│   │
│   └── evaluation/                 # 評価システム
│       ├── local.py                # ローカル評価
│       ├── langsmith.py            # LangSmith評価
│       └── dataset.py              # データセット作成
│
├── data/
│   ├── documents/                  # 検索対象ドキュメント
│   │   ├── 約款等/                 # 電気需給約款など
│   │   ├── 配送約款等/             # 託送供給約款など
│   │   └── system_manual/          # システムマニュアル
│   └── test_cases.csv              # 評価用Q&A
│
├── evaluation_results/             # 評価結果CSV
├── claude_output/                  # 分析ドキュメント出力
│
├── .chroma_db_agentic/             # ベクトルインデックス（自動生成）
├── .chroma_db_llamaindex/          # LlamaIndex用インデックス（自動生成）
└── .bm25_index_agentic.pkl         # BM25インデックス（自動生成）
```

## 処理フロー

```
ユーザー質問
    │
    ▼
┌─────────────────────────────────┐
│  ReAct Agent (LangGraph)        │
│  └─ Tool: search_manual         │
│       │                         │
│       ▼                         │
│  ┌─────────────────────────┐    │
│  │  Hybrid Search          │    │
│  │  ├─ BM25 (MeCab)        │    │
│  │  └─ Vector (Embedding)  │    │
│  │       ↓                 │    │
│  │  RRF Fusion             │    │
│  │       ↓                 │    │
│  │  電圧フィルタ           │    │
│  └─────────────────────────┘    │
│       │                         │
│       ▼                         │
│  回答生成                       │
└─────────────────────────────────┘
    │
    ▼
ユーザーへの回答
```

## パフォーマンス

| 項目 | 目標 | 実績 |
|------|------|------|
| 応答時間 | < 5秒 | ~2秒 |
| 精度 (>=0.7) | > 90% | ~85% |

## 開発

### テストケース追加

`data/test_cases.csv`を編集後:

```bash
python scripts/run_create_dataset.py  # LangSmithに反映
python scripts/run_evaluate_local.py  # 評価実行
```

### インデックス再作成

ドキュメントを追加・変更した場合:

```bash
python scripts/run_index.py --force
```
