"""Promptキャッシングユーティリティ

LLMプロバイダー別のプロンプトキャッシング対応:
- Anthropic: cache_controlを明示的に設定
- OpenAI: 1024トークン以上で自動キャッシュ（特別な処理不要）

効果:
- キャッシュヒット時、入力トークンコストが最大90%削減
- 長いシステムプロンプトを繰り返し使う場合に特に有効
"""

from langchain_core.messages import SystemMessage

from src.config import settings


def create_cached_system_message(prompt: str) -> SystemMessage | str:
    """キャッシュ対応のシステムメッセージを作成

    Args:
        prompt: システムプロンプトのテキスト

    Returns:
        - Anthropic: cache_control付きSystemMessage
        - OpenAI/その他: プレーンなプロンプト文字列

    Example:
        >>> from src.utils import create_cached_system_message
        >>> cached_prompt = create_cached_system_message(SYSTEM_PROMPT)
        >>> agent = create_react_agent(llm, tools, prompt=cached_prompt)
    """
    # キャッシングが無効の場合はそのまま返す
    if not settings.PROMPT_CACHING_ENABLED:
        return prompt

    # プロバイダー判定（モデル名から推測）
    model = settings.OPENAI_MODEL.lower()

    if "claude" in model or _is_anthropic_model(model):
        # Anthropic: cache_controlを設定
        return SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        )

    # OpenAI: 1024トークン以上で自動キャッシュされるため特別な処理不要
    # そのままプロンプト文字列を返す
    return prompt


def _is_anthropic_model(model: str) -> bool:
    """Anthropicモデルかどうかを判定"""
    anthropic_patterns = [
        "claude",
        "anthropic",
    ]
    return any(pattern in model for pattern in anthropic_patterns)


def estimate_token_count(text: str) -> int:
    """テキストのトークン数を概算

    日本語は1文字≒1-2トークン、英語は1単語≒1トークンとして概算。
    正確なカウントにはtiktokenを使用することを推奨。

    Args:
        text: トークン数を概算するテキスト

    Returns:
        概算トークン数
    """
    # 日本語文字数をカウント
    japanese_chars = sum(1 for c in text if ord(c) > 0x3000)
    # 英語は単語数で概算
    english_words = len(text.split()) - japanese_chars // 2

    # 日本語: 1文字≒1.5トークン、英語: 1単語≒1トークン
    return int(japanese_chars * 1.5 + max(0, english_words))


__all__ = ["create_cached_system_message", "estimate_token_count"]
