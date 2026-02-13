"""
LlamaIndex ハイブリッド検索モジュール

BM25 + Vector検索をQueryFusionRetrieverで統合。
RRF（Reciprocal Rank Fusion）でランク統合を行う。
"""

import pickle
from pathlib import Path
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

from src.config import settings
from src.rag.tokenizer import get_tokenizer

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHROMA_PERSIST_DIR = PROJECT_ROOT / settings.CHROMA_PERSIST_DIR
BM25_INDEX_PATH = PROJECT_ROOT / ".bm25_index_llamaindex.pkl"

# グローバルキャッシュ
_index = None
_bm25_retriever = None
_vector_retriever = None


def reset_cache():
    """キャッシュをリセット"""
    global _index, _bm25_retriever, _vector_retriever
    _index = None
    _bm25_retriever = None
    _vector_retriever = None


def get_index() -> VectorStoreIndex:
    """ベクトルインデックスを取得"""
    global _index

    if _index is None:
        if not CHROMA_PERSIST_DIR.exists():
            raise FileNotFoundError(
                f"インデックスが見つかりません: {CHROMA_PERSIST_DIR}\n"
                "先に 'python scripts/run_index_llamaparse.py --force' を実行してください。"
            )

        chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        chroma_collection = chroma_client.get_collection("llamaindex_docs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        embed_model = OpenAIEmbedding(model=settings.EMBEDDING_MODEL)

        _index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

    return _index


def get_all_nodes():
    """ChromaDBから全ノードを取得してTextNodeに変換"""
    from llama_index.core.schema import TextNode

    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(f"インデックスが見つかりません: {CHROMA_PERSIST_DIR}")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = chroma_client.get_collection("llamaindex_docs")

    # 全ドキュメントを取得
    results = collection.get(include=["documents", "metadatas"])

    nodes = []
    for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
        node = TextNode(
            text=doc,
            metadata=metadata or {},
            id_=f"node_{i}",
        )
        nodes.append(node)

    return nodes


def get_bm25_retriever(k: int = 10) -> BM25Retriever:
    """BM25 Retrieverを取得"""
    global _bm25_retriever

    if _bm25_retriever is None or _bm25_retriever.similarity_top_k != k:
        nodes = get_all_nodes()
        tokenizer = get_tokenizer()

        _bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=k,
            tokenizer=tokenizer,
        )

    return _bm25_retriever


def get_vector_retriever(k: int = 10):
    """Vector Retrieverを取得"""
    global _vector_retriever

    if _vector_retriever is None or _vector_retriever.similarity_top_k != k:
        index = get_index()
        _vector_retriever = index.as_retriever(similarity_top_k=k)

    return _vector_retriever


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


def filter_by_voltage_type(
    nodes: list[NodeWithScore],
    voltage_type: str
) -> list[NodeWithScore]:
    """
    検索結果を電圧タイプでフィルタリング・優先順位付け

    - 一致するものを上位に
    - 電圧タイプがないものは中位に
    - 不一致のものは下位に
    """
    matched = []
    neutral = []
    unmatched = []

    for node_with_score in nodes:
        doc_voltage = node_with_score.node.metadata.get("voltage_type", "")

        if doc_voltage == voltage_type:
            matched.append(node_with_score)
        elif doc_voltage == "":
            neutral.append(node_with_score)
        else:
            # 高圧特別高圧は高圧・特別高圧どちらの質問にもマッチさせる
            if doc_voltage == "高圧特別高圧" and voltage_type in ["高圧", "特別高圧"]:
                matched.append(node_with_score)
            else:
                unmatched.append(node_with_score)

    return matched + neutral + unmatched


def rrf_fusion(
    results_list: list[list[NodeWithScore]],
    k: int = 60
) -> list[NodeWithScore]:
    """
    Reciprocal Rank Fusion（RRF）でランク統合

    Args:
        results_list: 複数の検索結果リスト
        k: RRFのパラメータ（デフォルト60）

    Returns:
        統合された検索結果
    """
    fusion_scores = {}
    node_map = {}

    for results in results_list:
        for rank, node_with_score in enumerate(results):
            node_id = node_with_score.node.node_id
            rrf_score = 1 / (k + rank + 1)

            if node_id not in fusion_scores:
                fusion_scores[node_id] = 0
                node_map[node_id] = node_with_score

            fusion_scores[node_id] += rrf_score

    # スコア順にソート
    sorted_ids = sorted(fusion_scores.keys(), key=lambda x: fusion_scores[x], reverse=True)

    return [
        NodeWithScore(
            node=node_map[node_id].node,
            score=fusion_scores[node_id]
        )
        for node_id in sorted_ids
    ]


def hybrid_search(query: str, k: int = 5) -> list[NodeWithScore]:
    """
    ハイブリッド検索（BM25 + Vector）

    Args:
        query: 検索クエリ
        k: 最終的に返す件数

    Returns:
        検索結果のNodeWithScoreリスト
    """
    # 各Retrieverを取得（k*2件ずつ取得してRRF統合）
    bm25_retriever = get_bm25_retriever(k=k * 2)
    vector_retriever = get_vector_retriever(k=k * 2)

    query_bundle = QueryBundle(query_str=query)

    # 並列ではなく順次実行（シンプルさ優先）
    bm25_results = bm25_retriever.retrieve(query_bundle)
    vector_results = vector_retriever.retrieve(query_bundle)

    # RRF統合
    fused = rrf_fusion([bm25_results, vector_results])

    # 電圧タイプフィルタリング
    voltage_type = detect_voltage_type(query)
    if voltage_type:
        fused = filter_by_voltage_type(fused, voltage_type)

    return fused[:k]


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

    header = """【参照元】デジタルグリッド顧客ページ システムマニュアル・約款
※以下は当社サービスの公式ドキュメントからの抜粋です。

"""

    formatted_results = []
    for i, node_with_score in enumerate(results, 1):
        node = node_with_score.node
        metadata = node.metadata

        source = metadata.get("source", "不明")
        if "documents/" in source:
            source = source.split("documents/")[-1]

        doc_type = metadata.get("doc_type", "")
        section = metadata.get("section_title", "")
        area = metadata.get("area", "")
        voltage_type = metadata.get("voltage_type", "")

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

        content = node.get_content().strip()
        formatted_results.append(f"[{i}] {' | '.join(header_parts)}\n{content}")

    return header + "\n\n".join(formatted_results)


class HybridRetriever(BaseRetriever):
    """
    LlamaIndex用のカスタムハイブリッドRetriever

    QueryEngineやChatEngineと統合して使用可能。
    """

    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """検索を実行"""
        return hybrid_search(query_bundle.query_str, k=self.k)


def get_query_engine(k: int = 5):
    """
    ハイブリッド検索を使うQueryEngineを取得

    Args:
        k: 検索結果件数

    Returns:
        QueryEngine
    """
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.llms.openai import OpenAI

    retriever = HybridRetriever(k=k)
    llm = OpenAI(model=settings.OPENAI_MODEL, temperature=0.3)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
    )


if __name__ == "__main__":
    import time

    print("=== LlamaIndex ハイブリッドRAG検索テスト ===\n")

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
