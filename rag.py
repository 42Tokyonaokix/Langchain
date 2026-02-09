"""RAG (Retrieval-Augmented Generation) モジュール"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ドキュメントフォルダのパス
DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Chroma の永続化ディレクトリ
CHROMA_PERSIST_DIR = Path(__file__).parent / ".chroma_db"


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
    )
    try:
        all_docs.extend(pdf_loader.load())
    except Exception as e:
        print(f"PDF読み込みエラー: {e}")

    return all_docs


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """ドキュメントをチャンクに分割"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def create_vectorstore(documents=None, persist=True):
    """ベクトルストアを作成または読み込み"""
    embeddings = OpenAIEmbeddings()

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


def search_documents(query: str, k: int = 3) -> str:
    """ドキュメントを検索して結果を返す"""
    vectorstore = create_vectorstore()
    retriever = get_retriever(vectorstore, k=k)

    docs = retriever.invoke(query)

    if not docs:
        return "関連するドキュメントが見つかりませんでした。"

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "不明")
        content = doc.page_content.strip()
        results.append(f"[{i}] {source}\n{content}")

    return "\n\n".join(results)


if __name__ == "__main__":
    # テスト実行
    print("ドキュメントを読み込んでいます...")
    docs = load_documents()
    print(f"読み込んだドキュメント数: {len(docs)}")

    print("\nベクトルストアを作成しています...")
    vectorstore = create_vectorstore(docs)

    print("\n検索テスト: 'LangChain'")
    result = search_documents("LangChain")
    print(result)
