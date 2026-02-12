# デジタルグリッド RAGチャットボット

電力約款・システムマニュアルに基づいて回答するカスタマーサポートAIです。

## 特徴

- **高速検索**: ChromaDBによるベクトル検索（応答時間 < 1秒）
- **高精度**: 公式ドキュメントに基づく正確な回答
- **LangSmith対応**: 評価・モニタリング機能

## セットアップ

### 1. 環境構築

```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージインストール
pip install -r requirements.txt
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

`documents/`フォルダに検索対象のドキュメントを配置:

```
documents/
├── 約款等/
│   ├── 電気需給約款.pdf
│   └── ...
├── system_manual/
│   └── システムマニュアル.pdf
└── qa_list.csv
```

### 4. インデックス作成

```bash
python run_index.py --force
```

※ 初回のみ実行（約1分）。ドキュメント更新時は`--force`で再作成。

## 使い方

### チャットボット起動

```bash
python run_chatbot.py
```

### 評価実行

```bash
# テストケース更新（LangSmith）
python run_create_dataset.py

# 評価実行
python run_evaluate.py
```

## ファイル構成

```
.
├── run_chatbot.py        # チャットボット実行
├── run_create_dataset.py # LangSmithデータセット作成
├── run_evaluate.py       # 評価実行
├── run_index.py          # インデックス作成
├── test_cases.csv        # テストケース（101件）
│
├── documents/            # 検索対象ドキュメント
├── .chroma_db/           # ベクトルインデックス（自動生成）
│
└── src/
    ├── main/
    │   └── chatbot.py    # チャットボット本体
    └── utils/
        ├── rag.py        # RAG検索モジュール
        ├── index.py      # インデックス作成
        ├── create_dataset.py  # データセット作成
        └── evaluate.py   # 評価スクリプト
```

## パフォーマンス

| 項目 | 目標 | 実績 |
|------|------|------|
| 応答時間 | < 5秒 | ~1秒 |
| 精度 | > 90% | 評価中 |

## 開発

### テストケース追加

`test_cases.csv`を編集後:

```bash
python run_create_dataset.py  # LangSmithに反映
python run_evaluate.py        # 評価実行
```

### インデックス再作成

ドキュメントを追加・変更した場合:

```bash
python run_index.py --force
```
