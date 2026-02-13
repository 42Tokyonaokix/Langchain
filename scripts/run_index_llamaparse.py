#!/usr/bin/env python
"""
LlamaIndex インデックス作成スクリプト

LlamaParse + LlamaIndex IngestionPipelineを使用してインデックスを作成。

使用方法:
    # 通常実行
    python scripts/run_index_llamaparse.py --force

    # PDFを1件だけテスト
    python scripts/run_index_llamaparse.py --force --max=1

必要な環境変数:
    LLAMA_CLOUD_API_KEY: LlamaCloud APIキー
    OPENAI_API_KEY: OpenAI APIキー（Embedding用）
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.llamaindex_pipeline import create_llamaindex_index

if __name__ == "__main__":
    force = "--force" in sys.argv
    max_pdfs = None

    for arg in sys.argv[1:]:
        if arg.startswith("--max="):
            max_pdfs = int(arg.split("=")[1])

    create_llamaindex_index(force=force, max_pdfs=max_pdfs)
