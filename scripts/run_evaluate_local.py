#!/usr/bin/env python
"""
ローカル評価実行スクリプト

使用方法:
    # 全件評価
    python scripts/run_evaluate_local.py

    # 件数制限（テスト用）
    python scripts/run_evaluate_local.py --max=10

    # 出力ファイル名指定
    python scripts/run_evaluate_local.py --output=my_test

    # 組み合わせ
    python scripts/run_evaluate_local.py --max=20 --output=quick_test
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.local import main

if __name__ == "__main__":
    main()
