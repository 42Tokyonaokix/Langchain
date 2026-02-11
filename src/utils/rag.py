"""RAG (Retrieval-Augmented Generation) モジュール"""

import os
import pickle
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Optional
from pydantic import Field
from ddgs import DDGS

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ドキュメントフォルダのパス
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

# Chroma の永続化ディレクトリ
CHROMA_PERSIST_DIR = PROJECT_ROOT / ".chroma_db"

# チャンクキャッシュの保存先
CHUNKS_CACHE_FILE = PROJECT_ROOT / ".chunks_cache.pkl"


class EnsembleRetriever(BaseRetriever):
    """複数のRetrieverを重み付けで統合するRetriever"""

    retrievers: List[BaseRetriever] = Field(default_factory=list)
    weights: List[float] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """複数のRetrieverから検索し、スコアを統合して返す"""
        doc_scores = {}  # doc_id -> (doc, score)

        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                # Reciprocal Rank Fusion (RRF) スコアリング
                rrf_score = weight / (rank + 1)
                doc_id = doc.page_content[:100]  # コンテンツの先頭をIDとして使用

                if doc_id in doc_scores:
                    doc_scores[doc_id] = (doc, doc_scores[doc_id][1] + rrf_score)
                else:
                    doc_scores[doc_id] = (doc, rrf_score)

        # スコア順にソート
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs]


def load_documents():
    """documents/ フォルダからドキュメントを読み込む（txt, pdf対応）"""
    if not DOCUMENTS_DIR.exists():
        DOCUMENTS_DIR.mkdir(parents=True)
        return []

    all_docs = []

    # テキストファイルを読み込み
    txt_loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    try:
        all_docs.extend(txt_loader.load())
    except Exception as e:
        print(f"テキストファイル読み込みエラー: {e}")

    # PDFファイルを読み込み
    pdf_loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=False,
    )
    try:
        all_docs.extend(pdf_loader.load())
    except Exception as e:
        print(f"PDF読み込みエラー: {e}")

    # CSVファイルを読み込み
    csv_loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="**/*.csv",
        loader_cls=CSVLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    try:
        all_docs.extend(csv_loader.load())
    except Exception as e:
        print(f"CSV読み込みエラー: {e}")

    return all_docs


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """ドキュメントをチャンクに分割

    Args:
        documents: 分割対象のドキュメント
        chunk_size: チャンクサイズ（デフォルト1000文字）
        chunk_overlap: オーバーラップ（デフォルト100文字）
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def create_vectorstore(documents=None, persist=True):
    """ベクトルストアを作成または読み込み"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 永続化ディレクトリが存在し、ドキュメントが指定されていない場合は読み込み
    if persist and CHROMA_PERSIST_DIR.exists() and documents is None:
        return Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings,
        )

    # ドキュメントがない場合は読み込み
    if documents is None:
        documents = load_documents()

    if not documents:
        print("警告: ドキュメントが見つかりません。documents/ フォルダにファイルを追加してください。")
        # 空のベクトルストアを返す
        return Chroma(embedding_function=embeddings)

    # ドキュメントを分割
    chunks = split_documents(documents)
    print(f"{len(documents)} 個のドキュメントを {len(chunks)} 個のチャンクに分割しました。")

    # ベクトルストアを作成
    if persist:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
        )

    return vectorstore


def get_retriever(vectorstore=None, k=3):
    """検索用 Retriever を取得"""
    if vectorstore is None:
        vectorstore = create_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# キャッシュ用グローバル変数
_cached_chunks = None
_cached_bm25_retriever = None


def _get_documents_hash():
    """documentsフォルダの状態からハッシュを生成（変更検知用）"""
    if not DOCUMENTS_DIR.exists():
        return ""
    files = sorted(DOCUMENTS_DIR.rglob("*"))
    hash_input = ""
    for f in files:
        if f.is_file():
            stat = f.stat()
            hash_input += f"{f.name}:{stat.st_size}:{stat.st_mtime};"
    return hashlib.md5(hash_input.encode()).hexdigest()


def get_chunks(force_reload=False):
    """ドキュメントのチャンクを取得（永続化キャッシュ付き）

    Args:
        force_reload: Trueの場合、キャッシュを無視して再読込

    Returns:
        チャンクのリスト
    """
    global _cached_chunks

    # メモリキャッシュがあれば使用
    if _cached_chunks is not None and not force_reload:
        return _cached_chunks

    current_hash = _get_documents_hash()

    # ファイルキャッシュを確認
    if CHUNKS_CACHE_FILE.exists() and not force_reload:
        try:
            with open(CHUNKS_CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)
            if cache_data.get("hash") == current_hash:
                _cached_chunks = cache_data["chunks"]
                print(f"チャンクをキャッシュから読み込みました（{len(_cached_chunks)}件）")
                return _cached_chunks
        except Exception as e:
            print(f"キャッシュ読み込みエラー: {e}")

    # キャッシュがない or 古い場合は再生成
    print("ドキュメントを読み込んでチャンクを生成中...")
    documents = load_documents()
    if documents:
        _cached_chunks = split_documents(documents)
    else:
        _cached_chunks = []

    # ファイルに保存
    try:
        with open(CHUNKS_CACHE_FILE, "wb") as f:
            pickle.dump({"hash": current_hash, "chunks": _cached_chunks}, f)
        print(f"チャンクをキャッシュに保存しました（{len(_cached_chunks)}件）")
    except Exception as e:
        print(f"キャッシュ保存エラー: {e}")

    return _cached_chunks


def clear_chunk_cache():
    """チャンクキャッシュをクリア（ドキュメント更新時に呼び出す）"""
    global _cached_chunks, _cached_bm25_retriever
    _cached_chunks = None
    _cached_bm25_retriever = None
    if CHUNKS_CACHE_FILE.exists():
        CHUNKS_CACHE_FILE.unlink()
        print("チャンクキャッシュを削除しました")


def get_bm25_retriever(k=3):
    """BM25キーワード検索用Retrieverを取得（キャッシュ付き）"""
    global _cached_bm25_retriever

    chunks = get_chunks()
    if not chunks:
        return None

    # キャッシュがあれば再利用（kだけ更新）
    if _cached_bm25_retriever is not None:
        _cached_bm25_retriever.k = k
        return _cached_bm25_retriever

    _cached_bm25_retriever = BM25Retriever.from_documents(chunks, k=k)
    return _cached_bm25_retriever


def get_hybrid_retriever(vectorstore=None, k=3, vector_weight=0.5, bm25_weight=0.5):
    """ハイブリッド検索用Retrieverを取得

    Args:
        vectorstore: ベクトルストア（Noneの場合は自動作成）
        k: 各Retrieverから取得する件数
        vector_weight: ベクトル検索の重み（0.0-1.0）
        bm25_weight: BM25検索の重み（0.0-1.0）

    Returns:
        EnsembleRetriever（ハイブリッド検索用）
    """
    # ベクトル検索用Retriever
    vector_retriever = get_retriever(vectorstore, k=k)

    # BM25キーワード検索用Retriever
    bm25_retriever = get_bm25_retriever(k=k)

    if bm25_retriever is None:
        print("警告: BM25 Retrieverを作成できませんでした。ベクトル検索のみ使用します。")
        return vector_retriever

    # EnsembleRetrieverで統合
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )

    return ensemble_retriever


def _score_single_doc(args):
    """単一ドキュメントのスコアリング（並列処理用）"""
    query, doc, llm = args
    content = doc.page_content[:500]

    prompt = f"""質問に対するドキュメントの関連性を0-10で評価してください。
数値のみを返してください。

質問: {query}

ドキュメント:
{content}

スコア:"""

    try:
        response = llm.invoke(prompt)
        score = float(response.content.strip())
    except (ValueError, Exception):
        score = 5.0

    return (score, doc)


def rerank_documents(query: str, docs: list, top_k: int = 5, max_workers: int = 5) -> list:
    """LLMを使って検索結果をリランキング（並列処理）

    Args:
        query: 検索クエリ
        docs: 検索結果のドキュメントリスト
        top_k: 返す上位件数
        max_workers: 並列処理のワーカー数

    Returns:
        関連性の高い順にソートされたドキュメントリスト
    """
    if not docs or len(docs) <= top_k:
        return docs

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 並列でスコアリング
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        args_list = [(query, doc, llm) for doc in docs]
        scored_docs = list(executor.map(_score_single_doc, args_list))

    # スコア順にソートして上位を返す
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k]]


def search_documents(
    query: str,
    k: int = 5,
    use_rerank: bool = True,
    use_hybrid: bool = True,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> str:
    """ドキュメントを検索して結果を返す

    Args:
        query: 検索クエリ
        k: 返す件数
        use_rerank: リランキングを使用するか
        use_hybrid: ハイブリッド検索を使用するか（ベクトル + BM25）
        vector_weight: ベクトル検索の重み（use_hybrid=Trueの場合）
        bm25_weight: BM25検索の重み（use_hybrid=Trueの場合）
    """
    vectorstore = create_vectorstore()

    # リランキングを使う場合は多めに検索
    search_k = k * 2 if use_rerank else k

    # Retrieverの選択
    if use_hybrid:
        retriever = get_hybrid_retriever(
            vectorstore,
            k=search_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
    else:
        retriever = get_retriever(vectorstore, k=search_k)

    docs = retriever.invoke(query)

    if not docs:
        return "関連するドキュメントが見つかりませんでした。"

    # リランキング
    if use_rerank:
        docs = rerank_documents(query, docs, top_k=k)

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "不明")
        content = doc.page_content.strip()
        results.append(f"[{i}] {source}\n{content}")

    return "\n\n".join(results)


def web_search(query: str, k: int = 5, region: str = "jp-jp") -> str:
    """DuckDuckGoでWeb検索を実行

    Args:
        query: 検索クエリ
        k: 返す件数
        region: 検索地域（デフォルト: 日本）

    Returns:
        検索結果の文字列
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region=region, max_results=k))

        if not results:
            return "Web検索結果が見つかりませんでした。"

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "タイトルなし")
            body = r.get("body", "")
            url = r.get("href", "")
            formatted.append(f"[{i}] {title}\n{body}\nURL: {url}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Web検索エラー: {e}"


def search_with_web_fallback(
    query: str,
    k: int = 5,
    use_rerank: bool = True,
    use_hybrid: bool = True,
    web_fallback: bool = True,
    min_docs_threshold: int = 2,
) -> str:
    """ドキュメント検索を行い、結果が少ない場合はWeb検索で補完

    Args:
        query: 検索クエリ
        k: 返す件数
        use_rerank: リランキングを使用するか
        use_hybrid: ハイブリッド検索を使用するか
        web_fallback: 結果が少ない場合にWeb検索を使うか
        min_docs_threshold: Web検索を発動する閾値（これ未満ならWeb検索）

    Returns:
        検索結果（ドキュメント + Web）
    """
    # まずローカルドキュメントを検索
    doc_result = search_documents(
        query=query,
        k=k,
        use_rerank=use_rerank,
        use_hybrid=use_hybrid,
    )

    # 結果を解析
    doc_count = doc_result.count("[") if doc_result != "関連するドキュメントが見つかりませんでした。" else 0

    results = ["【ドキュメント検索結果】", doc_result]

    # 結果が少ない場合はWeb検索で補完
    if web_fallback and doc_count < min_docs_threshold:
        web_result = web_search(query, k=k)
        results.append("\n【Web検索結果（補完）】")
        results.append(web_result)

    return "\n".join(results)


def combined_search(
    query: str,
    k: int = 5,
    use_rerank: bool = True,
    use_hybrid: bool = True,
    include_web: bool = False,
    web_k: int = 3,
) -> str:
    """ドキュメント検索とWeb検索を統合

    Args:
        query: 検索クエリ
        k: ドキュメント検索で返す件数
        use_rerank: リランキングを使用するか
        use_hybrid: ハイブリッド検索を使用するか
        include_web: Web検索も含めるか
        web_k: Web検索で返す件数

    Returns:
        統合された検索結果
    """
    results = []

    # ドキュメント検索
    doc_result = search_documents(
        query=query,
        k=k,
        use_rerank=use_rerank,
        use_hybrid=use_hybrid,
    )
    results.append("【ドキュメント検索結果】")
    results.append(doc_result)

    # Web検索（オプション）
    if include_web:
        web_result = web_search(query, k=web_k)
        results.append("\n【Web検索結果】")
        results.append(web_result)

    return "\n".join(results)


if __name__ == "__main__":
    # テスト実行
    print("ドキュメントを読み込んでいます...")
    docs = load_documents()
    print(f"読み込んだドキュメント数: {len(docs)}")

    print("\nベクトルストアを作成しています...")
    vectorstore = create_vectorstore(docs)

    print("\n=== ベクトル検索のみ ===")
    result = search_documents("LangChain", use_hybrid=False, use_rerank=False)
    print(result)

    print("\n=== ハイブリッド検索（ベクトル + BM25） ===")
    result = search_documents("LangChain", use_hybrid=True, use_rerank=False)
    print(result)

    print("\n=== ハイブリッド検索 + リランキング ===")
    result = search_documents("LangChain", use_hybrid=True, use_rerank=True)
    print(result)

    print("\n=== Web検索 ===")
    result = web_search("電力自由化 最新情報", k=3)
    print(result)
