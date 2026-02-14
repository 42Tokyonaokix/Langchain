# ベースイメージ: Python 3.11（軽量版）
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# 依存パッケージ定義を先にコピー（キャッシュ効率化）
COPY requirements.txt .

# パッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# Python出力をバッファリングしない（ログがリアルタイムで見える）
ENV PYTHONUNBUFFERED=1

# 起動コマンド（src/main/chatbot.py を実行）
CMD ["python", "-m", "src.main.chatbot"]
