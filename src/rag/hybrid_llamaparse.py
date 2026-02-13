"""
LlamaParse対応 ハイブリッドRAGモジュール

LlamaParseで作成したインデックスを使用。
表データを含むチャンクも適切に検索。
"""

import os
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
import MeCab

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHROMA_PERSIST_DIR = PROJECT_ROOT / ".chroma_db_llamaparse"
BM25_INDEX_PATH = PROJECT_ROOT / ".bm25_index_llamaparse.pkl"

# グローバルキャッシュ
_embeddings = None
_vectorstore = None
_bm25_index = None
_bm25_docs = None
_mecab = None


def reset_cache():
    """キャッシュをリセット"""
    global _embeddings, _vectorstore, _bm25_index, _bm25_docs
    _embeddings = None
    _vectorstore = None
    _bm25_index = None
    _bm25_docs = None


def get_mecab():
    """MeCabトークナイザーを取得"""
    global _mecab
    if _mecab is None:
        _mecab = MeCab.Tagger("-Owakati")
    return _mecab


def tokenize(text: str) -> list[str]:
    """日本語テキストをトークン化"""
    mecab = get_mecab()
    return mecab.parse(text).strip().split()


def detect_voltage_type(query: str) -> Optional[str]:
    """クエリから電圧タイプを検出"""
    if "高圧特別高圧" in query or ("高圧" in query and "特別高圧" in query):
        return "高圧特別高圧"
    if "特別高圧" in query:
        return "特別高圧"
    if "高圧" in query and "低圧" not in query:
        return "高圧"
    if "低圧" in query:
        return "低圧"
    return None


def filter_by_voltage_type(results: list[dict], voltage_type: str) -> list[dict]:
    """検索結果を電圧タイプでフィルタリング"""
    matched = []
    neutral = []
    unmatched = []

    for result in results:
        doc_voltage = result.get("metadata", {}).get("voltage_type", "")

        if doc_voltage == voltage_type:
            matched.append(result)
        elif doc_voltage == "" or doc_voltage is None:
            neutral.append(result)
        else:
            if doc_voltage == "高圧特別高圧" and voltage_type in ["高圧", "特別高圧"]:
                matched.append(result)
            else:
                unmatched.append(result)

    return matched + neutral + unmatched


def get_embeddings():
    """Embeddingsを取得"""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings


def get_vectorstore():
    """ベクトルストアを取得"""
    global _vectorstore
    if _vectorstore is None:
        if not CHROMA_PERSIST_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDBが見つかりません: {CHROMA_PERSIST_DIR}\n"
                "先に 'python scripts/run_index_llamaparse.py --force' を実行してください。"
            )
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=get_embeddings(),
        )
    return _vectorstore


def get_bm25_index():
    """BM25インデックスを取得"""
    global _bm25_index, _bm25_docs

    if _bm25_index is None:
        if BM25_INDEX_PATH.exists():
            with open(BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                _bm25_index = data["index"]
                _bm25_docs = data["docs"]
        else:
            _bm25_index, _bm25_docs = build_bm25_index()

    return _bm25_index, _bm25_docs


def build_bm25_index():
    """BM25インデックスを構築"""
    print("BM25インデックスを構築中...")

    vectorstore = get_vectorstore()
    collection = vectorstore._collection

    results = collection.get(include=["documents", "metadatas"])
    docs = results["documents"]
    metadatas = results["metadatas"]

    tokenized_docs = [tokenize(doc) for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)

    doc_info = [
        {"content": doc, "metadata": meta}
        for doc, meta in zip(docs, metadatas)
    ]

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"index": bm25, "docs": doc_info}, f)

    print(f"BM25インデックスを保存: {BM25_INDEX_PATH}")
    print(f"ドキュメント数: {len(docs)}")

    return bm25, doc_info


def search_bm25(query: str, k: int = 10) -> list[dict]:
    """BM25検索"""
    bm25, docs = get_bm25_index()
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": docs[idx]["content"],
                "metadata": docs[idx]["metadata"],
                "score": scores[idx],
                "source": "bm25"
            })

    return results


def search_vector(query: str, k: int = 10) -> list[dict]:
    """ベクトル検索"""
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": 1 / (1 + score),
            "source": "vector"
        }
        for doc, score in results
    ]


def rrf_fusion(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    """RRF統合"""
    fusion_scores = {}
    doc_info = {}

    for results in results_list:
        for rank, result in enumerate(results):
            content = result["content"]
            rrf_score = 1 / (k + rank + 1)

            if content not in fusion_scores:
                fusion_scores[content] = 0
                doc_info[content] = result

            fusion_scores[content] += rrf_score

    sorted_contents = sorted(fusion_scores.keys(), key=lambda x: fusion_scores[x], reverse=True)

    return [
        {**doc_info[content], "fusion_score": fusion_scores[content]}
        for content in sorted_contents
    ]


def hybrid_search(query: str, k: int = 5) -> list[dict]:
    """ハイブリッド検索（BM25 + Vector）"""
    results_bm25 = []
    results_vector = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_bm25 = executor.submit(search_bm25, query, k * 2)
        future_vector = executor.submit(search_vector, query, k * 2)

        results_bm25 = future_bm25.result()
        results_vector = future_vector.result()

    if results_bm25 or results_vector:
        fused = rrf_fusion([results_bm25, results_vector])

        voltage_type = detect_voltage_type(query)
        if voltage_type:
            fused = filter_by_voltage_type(fused, voltage_type)

        return fused[:k]

    return []


def search_with_context(query: str, k: int = 5) -> str:
    """検索結果を文脈情報付きで返す"""
    results = hybrid_search(query, k=k)

    if not results:
        return "関連するドキュメントが見つかりませんでした。"

    header = """【参照元】デジタルグリッド顧客ページ システムマニュアル・約款
※以下は当社サービスの公式ドキュメントからの抜粋です。

"""

    formatted_results = []
    for i, result in enumerate(results, 1):
        source = result["metadata"].get("source", "不明")
        if "documents/" in source:
            source = source.split("documents/")[-1]

        doc_type = result["metadata"].get("doc_type", "")
        section = result["metadata"].get("section_title", "")
        area = result["metadata"].get("area", "")
        voltage_type = result["metadata"].get("voltage_type", "")
        has_table = result["metadata"].get("has_table", False)

        header_parts = [f"出典: {source}"]
        if doc_type:
            header_parts.append(f"種別: {doc_type}")
        if voltage_type:
            header_parts.append(f"電圧: {voltage_type}")
        if area:
            header_parts.append(f"エリア: {area}")
        if section:
            header_parts.append(f"セクション: {section}")
        if has_table:
            header_parts.append("📊表あり")

        content = result["content"].strip()
        formatted_results.append(f"[{i}] {' | '.join(header_parts)}\n{content}")

    return header + "\n\n".join(formatted_results)


if __name__ == "__main__":
    import time

    print("=== LlamaParse Hybrid RAG テスト ===\n")

    if not BM25_INDEX_PATH.exists():
        build_bm25_index()

    test_queries = [
        "料金表",
        "契約電力の算定方法",
        "託送供給等約款の適用",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"クエリ: {query}")
        print('='*50)

        start = time.time()
        result = search_with_context(query, k=3)
        elapsed = time.time() - start

        print(f"検索時間: {elapsed:.2f}秒")
        print(result[:1500])
