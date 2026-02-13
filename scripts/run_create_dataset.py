#!/usr/bin/env python
"""データセット作成スクリプト"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.dataset import main

if __name__ == "__main__":
    main()
