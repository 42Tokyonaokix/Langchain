#!/usr/bin/env python
"""評価実行スクリプト（LangSmith）"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.langsmith import main

if __name__ == "__main__":
    main()
