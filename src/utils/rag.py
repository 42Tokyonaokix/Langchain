"""
高速RAGモジュール - 精度90%以上、5秒以内を目標

設計方針:
- 検索時はChromaDBからの読み込みのみ（PDF読み込みなし）
- インデックス作成は事前に1回だけ（index.pyで実行）
- シンプルなベクトル検索
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHROMA_PERSIST_DIR = PROJECT_ROOT / ".chroma_db"

# Embeddingsはグローバルで1回だけ初期化
_embeddings = None
_vectorstore = None


def get_embeddings():
    """Embeddingsを取得（シングルトン）"""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings


def get_vectorstore():
    """ベクトルストアを取得（シングルトン）"""
    global _vectorstore
    if _vectorstore is None:
        if not CHROMA_PERSIST_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDBが見つかりません: {CHROMA_PERSIST_DIR}\n"
                "先に 'python src/utils/index.py' を実行してインデックスを作成してください。"
            )
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=get_embeddings(),
        )
    return _vectorstore


def search(query: str, k: int = 5) -> list:
    """
    ベクトル検索を実行

    Args:
        query: 検索クエリ
        k: 取得件数

    Returns:
        検索結果のDocumentリスト
    """
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=k)


def search_with_context(query: str, k: int = 5) -> str:
    """
    検索結果を文脈情報付きの文字列で返す

    Args:
        query: 検索クエリ
        k: 取得件数

    Returns:
        フォーマット済みの検索結果文字列
    """
    docs = search(query, k=k)

    if not docs:
        return "関連するドキュメントが見つかりませんでした。"

    # 文脈情報を追加
    header = """【参照元】デジタルグリッド顧客ページ システムマニュアル・約款
※以下は当社サービスの公式ドキュメントからの抜粋です。

"""

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "不明")
        # パスを簡略化
        if "documents/" in source:
            source = source.split("documents/")[-1]

        # メタデータからセクション・ドキュメントタイプを取得
        doc_type = doc.metadata.get("doc_type", "")
        section = doc.metadata.get("section", "")

        # ヘッダー行を構築
        header_parts = [f"出典: {source}"]
        if doc_type:
            header_parts.append(f"種別: {doc_type}")
        if section:
            header_parts.append(f"セクション: {section}")

        content = doc.page_content.strip()
        results.append(f"[{i}] {' | '.join(header_parts)}\n{content}")

    return header + "\n\n".join(results)


if __name__ == "__main__":
    import time

    print("=== RAG検索テスト ===")

    start = time.time()
    result = search_with_context("契約期間", k=3)
    elapsed = time.time() - start

    print(f"検索時間: {elapsed:.2f}秒")
    print()
    print(result[:1000])
