"""
ハイブリッドRAGモジュール - BM25 + Vector検索の並列処理

特徴:
- BM25（キーワード検索）とVector検索を並列実行
- 早期終了: 高スコアの結果があれば即座に返す
- RRF（Reciprocal Rank Fusion）でランク統合
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
# Agenticインデックスを使用（より高精度な構造化分割）
CHROMA_PERSIST_DIR = PROJECT_ROOT / ".chroma_db_agentic"
BM25_INDEX_PATH = PROJECT_ROOT / ".bm25_index_agentic.pkl"

# グローバルキャッシュ（インデックス切り替え時はNoneにリセット）
_embeddings = None
_vectorstore = None
_bm25_index = None
_bm25_docs = None
_mecab = None


def reset_cache():
    """キャッシュをリセット（インデックス切り替え時に使用）"""
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
    """
    クエリから電圧タイプを検出

    Returns:
        "高圧特別高圧", "特別高圧", "高圧", "低圧", or None
    """
    # 優先順位順にチェック（より具体的なものを先に）
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
    """
    検索結果を電圧タイプでフィルタリング・優先順位付け

    - 一致するものを上位に
    - 電圧タイプがないものは中位に
    - 不一致のものは下位に
    """
    matched = []
    neutral = []  # voltage_typeがないドキュメント
    unmatched = []

    for result in results:
        doc_voltage = result.get("metadata", {}).get("voltage_type", "")

        if doc_voltage == voltage_type:
            matched.append(result)
        elif doc_voltage == "":
            neutral.append(result)
        else:
            # 高圧特別高圧は高圧・特別高圧どちらの質問にもマッチさせる
            if doc_voltage == "高圧特別高圧" and voltage_type in ["高圧", "特別高圧"]:
                matched.append(result)
            else:
                unmatched.append(result)

    return matched + neutral + unmatched


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


def get_bm25_index():
    """BM25インデックスを取得（シングルトン）"""
    global _bm25_index, _bm25_docs

    if _bm25_index is None:
        if BM25_INDEX_PATH.exists():
            # 既存のインデックスを読み込み
            with open(BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                _bm25_index = data["index"]
                _bm25_docs = data["docs"]
        else:
            # インデックスを新規作成
            _bm25_index, _bm25_docs = build_bm25_index()

    return _bm25_index, _bm25_docs


def build_bm25_index():
    """BM25インデックスを構築"""
    print("BM25インデックスを構築中...")

    vectorstore = get_vectorstore()
    collection = vectorstore._collection

    # 全ドキュメントを取得
    results = collection.get(include=["documents", "metadatas"])
    docs = results["documents"]
    metadatas = results["metadatas"]

    # トークン化
    tokenized_docs = [tokenize(doc) for doc in docs]

    # BM25インデックス作成
    bm25 = BM25Okapi(tokenized_docs)

    # ドキュメント情報を保持
    doc_info = [
        {"content": doc, "metadata": meta}
        for doc, meta in zip(docs, metadatas)
    ]

    # インデックスを保存
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"index": bm25, "docs": doc_info}, f)

    print(f"BM25インデックスを保存しました: {BM25_INDEX_PATH}")
    print(f"ドキュメント数: {len(docs)}")

    return bm25, doc_info


def search_bm25(query: str, k: int = 10) -> list[dict]:
    """BM25検索を実行"""
    bm25, docs = get_bm25_index()

    # クエリをトークン化
    tokenized_query = tokenize(query)

    # スコア計算
    scores = bm25.get_scores(tokenized_query)

    # 上位k件を取得
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # スコアが0より大きいもののみ
            results.append({
                "content": docs[idx]["content"],
                "metadata": docs[idx]["metadata"],
                "score": scores[idx],
                "source": "bm25"
            })

    return results


def search_vector(query: str, k: int = 10) -> list[dict]:
    """ベクトル検索を実行"""
    vectorstore = get_vectorstore()

    # スコア付きで検索
    results = vectorstore.similarity_search_with_score(query, k=k)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": 1 / (1 + score),  # 距離をスコアに変換（0-1の範囲）
            "source": "vector"
        }
        for doc, score in results
    ]


def rrf_fusion(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion（RRF）でランク統合

    Args:
        results_list: 複数の検索結果リスト
        k: RRFのパラメータ（デフォルト60）

    Returns:
        統合された検索結果
    """
    # コンテンツをキーにしてスコアを集計
    fusion_scores = {}
    doc_info = {}

    for results in results_list:
        for rank, result in enumerate(results):
            content = result["content"]
            # RRFスコア計算: 1 / (k + rank)
            rrf_score = 1 / (k + rank + 1)

            if content not in fusion_scores:
                fusion_scores[content] = 0
                doc_info[content] = result

            fusion_scores[content] += rrf_score

    # スコア順にソート
    sorted_contents = sorted(fusion_scores.keys(), key=lambda x: fusion_scores[x], reverse=True)

    return [
        {
            **doc_info[content],
            "fusion_score": fusion_scores[content]
        }
        for content in sorted_contents
    ]


def hybrid_search(query: str, k: int = 5, early_exit_threshold: float = 0.8) -> list[dict]:
    """
    ハイブリッド検索（BM25 + Vector）を並列実行

    Args:
        query: 検索クエリ
        k: 最終的に返す件数
        early_exit_threshold: 早期終了のスコア閾値

    Returns:
        検索結果リスト
    """
    results_bm25 = []
    results_vector = []
    early_result = None
    lock = threading.Lock()

    def run_bm25():
        nonlocal results_bm25, early_result
        results = search_bm25(query, k=k*2)
        with lock:
            results_bm25 = results
            # 早期終了チェック: BM25で高スコアがあれば
            if results and results[0]["score"] > early_exit_threshold * 10:  # BM25スコアは大きい
                early_result = results[:k]

    def run_vector():
        nonlocal results_vector, early_result
        results = search_vector(query, k=k*2)
        with lock:
            results_vector = results
            # 早期終了チェック: Vectorで高スコアがあれば
            if results and results[0]["score"] > early_exit_threshold:
                if early_result is None:  # BM25がまだ終わっていなければ
                    early_result = results[:k]

    # 並列実行
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_bm25),
            executor.submit(run_vector)
        ]

        # 最初に完了したものをチェック
        for future in as_completed(futures):
            future.result()
            # 早期終了チェック
            if early_result is not None:
                # 両方の結果が揃うまで待つ（RRF統合のため）
                pass

    # RRF統合
    if results_bm25 or results_vector:
        fused = rrf_fusion([results_bm25, results_vector])

        # 電圧タイプが質問に含まれている場合のみフィルタリング
        voltage_type = detect_voltage_type(query)
        if voltage_type:
            fused = filter_by_voltage_type(fused, voltage_type)

        return fused[:k]

    return []


def search_with_context(query: str, k: int = 5) -> str:
    """
    ハイブリッド検索結果を文脈情報付きの文字列で返す

    Args:
        query: 検索クエリ
        k: 取得件数

    Returns:
        フォーマット済みの検索結果文字列
    """
    results = hybrid_search(query, k=k)

    if not results:
        return "関連するドキュメントが見つかりませんでした。"

    # 文脈情報を追加
    header = """【参照元】デジタルグリッド顧客ページ システムマニュアル・約款
※以下は当社サービスの公式ドキュメントからの抜粋です。

"""

    formatted_results = []
    for i, result in enumerate(results, 1):
        source = result["metadata"].get("source", "不明")
        # パスを簡略化
        if "documents/" in source:
            source = source.split("documents/")[-1]

        # メタデータからセクション・ドキュメントタイプ・エリア・電圧タイプを取得
        doc_type = result["metadata"].get("doc_type", "")
        section = result["metadata"].get("section", "")
        area = result["metadata"].get("area", "")
        voltage_type = result["metadata"].get("voltage_type", "")
        search_source = result.get("source", "")

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
        if search_source:
            header_parts.append(f"検索: {search_source}")

        content = result["content"].strip()
        formatted_results.append(f"[{i}] {' | '.join(header_parts)}\n{content}")

    return header + "\n\n".join(formatted_results)


if __name__ == "__main__":
    import time

    print("=== ハイブリッドRAG検索テスト ===\n")

    # BM25インデックスを構築（初回のみ）
    if not BM25_INDEX_PATH.exists():
        build_bm25_index()

    # テストクエリ
    test_queries = [
        "スイッチングに必要な情報 供給地点特定番号 桁数",
        "学習用データの定義",
        "パスワードを忘れた場合",
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
