"""
高速RAGモジュール - 精度90%以上、5秒以内を目標

設計方針:
- 検索時はChromaDBからの読み込みのみ（PDF読み込みなし）
- インデックス作成は事前に1回だけ（index.pyで実行）
- シンプルなベクトル検索
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def detect_voltage_type(query: str) -> Optional[str]:
    """
    クエリから電圧タイプを検出

    Returns:
        "高圧特別高圧", "特別高圧", "高圧", "低圧", or None
    """
    if "高圧特別高圧" in query or ("高圧" in query and "特別高圧" in query):
        return "高圧特別高圧"
    if "特別高圧" in query:
        return "特別高圧"
    if "高圧" in query and "低圧" not in query:
        return "高圧"
    if "低圧" in query:
        return "低圧"
    return None


def filter_docs_by_voltage_type(docs: list, voltage_type: str) -> list:
    """
    検索結果を電圧タイプでフィルタリング・優先順位付け
    """
    matched = []
    neutral = []
    unmatched = []

    for doc in docs:
        doc_voltage = doc.metadata.get("voltage_type", "")

        if doc_voltage == voltage_type:
            matched.append(doc)
        elif doc_voltage == "":
            neutral.append(doc)
        else:
            if doc_voltage == "高圧特別高圧" and voltage_type in ["高圧", "特別高圧"]:
                matched.append(doc)
            else:
                unmatched.append(doc)

    return matched + neutral + unmatched

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
# Agenticインデックスを使用（より高精度な構造化分割）
CHROMA_PERSIST_DIR = PROJECT_ROOT / ".chroma_db_agentic"

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
                "先に 'python scripts/run_index.py' を実行してインデックスを作成してください。"
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

    # 電圧タイプが質問に含まれている場合、多めに取得してフィルタリング
    voltage_type = detect_voltage_type(query)
    if voltage_type:
        docs = vectorstore.similarity_search(query, k=k*2)
        docs = filter_docs_by_voltage_type(docs, voltage_type)
        return docs[:k]

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

        # メタデータからセクション・ドキュメントタイプ・エリア・電圧タイプを取得
        doc_type = doc.metadata.get("doc_type", "")
        section = doc.metadata.get("section", "")
        area = doc.metadata.get("area", "")
        voltage_type = doc.metadata.get("voltage_type", "")

        # ヘッダー行を構築
        header_parts = [f"出典: {source}"]
        if doc_type:
            header_parts.append(f"種別: {doc_type}")
        if voltage_type:
            header_parts.append(f"電圧: {voltage_type}")
        if area:
            header_parts.append(f"エリア: {area}")
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
