#!/usr/bin/env python
"""インデックス作成スクリプト"""
import sys
from src.utils.index import create_index

if __name__ == "__main__":
    force = "--force" in sys.argv
    create_index(force=force)
