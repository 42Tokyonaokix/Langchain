"""アプリケーション設定

環境変数を型安全に一元管理。
.envファイルから自動読み込み。
"""

from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# .envから環境変数をロード（LlamaIndex等が環境変数を直接参照するため）
load_dotenv()


class Settings(BaseSettings):
    """アプリケーション設定（環境変数から自動読み込み）"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 未定義の環境変数は無視
    )

    # ==================== LLM設定 ====================
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o"

    # ==================== LlamaParse設定 ====================
    LLAMA_CLOUD_API_KEY: str | None = None

    # ==================== LangSmith設定 ====================
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "dgm-chatbot"

    # ==================== RAG設定 ====================
    # ChromaDB
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_PERSIST_DIR: str = ".chroma_db_llamaindex"

    # 検索設定
    RAG_TOP_K: int = 5
    RAG_BM25_WEIGHT: float = 0.5
    RAG_VECTOR_WEIGHT: float = 0.5

    # チャンク設定
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 200

    # ==================== Embedding設定 ====================
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ==================== インフラ設定 ====================
    REDIS_URL: str = "redis://localhost:6379"
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/chatbot"

    # ==================== 機能フラグ ====================
    PROMPT_CACHING_ENABLED: bool = True
    PDF_VISION_ENABLED: bool = False
    PDF_VISION_MODEL: str = "gpt-4o"
    PDF_VISION_FALLBACK_THRESHOLD: int = 100  # ページ単位: この文字数以下でVisionにフォールバック

    # ==================== ログ設定 ====================
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


# シングルトンインスタンス（アプリ全体で共有）
settings = Settings()

__all__ = ["Settings", "settings"]
