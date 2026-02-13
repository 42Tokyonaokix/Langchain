#!/usr/bin/env python
"""インデックス作成スクリプト"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.basic import create_index

if __name__ == "__main__":
    force = "--force" in sys.argv
    create_index(force=force)
