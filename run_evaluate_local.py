#!/usr/bin/env python
"""
ローカル評価実行スクリプト

使用方法:
    # 全件評価
    python run_evaluate_local.py

    # 件数制限（テスト用）
    python run_evaluate_local.py --max=10

    # 出力ファイル名指定
    python run_evaluate_local.py --output=my_test

    # 組み合わせ
    python run_evaluate_local.py --max=20 --output=quick_test
"""
from src.utils.evaluate_local import main

if __name__ == "__main__":
    main()
