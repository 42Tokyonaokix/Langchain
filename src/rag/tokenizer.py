"""
日本語トークナイザー - MeCabベース

LlamaIndex BM25Retrieverで使用するための日本語トークナイザー。
"""

from typing import Callable

import MeCab


_mecab_tagger = None


def get_mecab_tagger() -> MeCab.Tagger:
    """MeCabタガーを取得（シングルトン）"""
    global _mecab_tagger
    if _mecab_tagger is None:
        _mecab_tagger = MeCab.Tagger("-Owakati")
    return _mecab_tagger


def mecab_tokenize(text: str) -> list[str]:
    """
    MeCabで日本語テキストをトークン化

    Args:
        text: トークン化するテキスト

    Returns:
        トークンのリスト
    """
    tagger = get_mecab_tagger()
    result = tagger.parse(text)
    if result is None:
        return []
    return result.strip().split()


def get_tokenizer() -> Callable[[str], list[str]]:
    """
    LlamaIndex BM25Retrieverで使用するトークナイザー関数を取得

    Returns:
        トークナイザー関数
    """
    return mecab_tokenize


if __name__ == "__main__":
    # テスト
    test_texts = [
        "電気需給約款に基づく契約条件について",
        "高圧の託送供給等約款を確認したい",
        "供給地点特定番号は22桁です",
    ]

    tokenizer = get_tokenizer()
    for text in test_texts:
        tokens = tokenizer(text)
        print(f"入力: {text}")
        print(f"トークン: {tokens}")
        print()
