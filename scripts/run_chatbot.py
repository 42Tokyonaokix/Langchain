#!/usr/bin/env python
"""チャットボット実行スクリプト"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chatbot.agent import main

if __name__ == "__main__":
    main()
