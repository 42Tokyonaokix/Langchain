"""
LlamaIndex IngestionPipeline

LlamaParse + カスタム抽出器を使ったインデックス作成パイプライン。
"""

import re
import shutil
from pathlib import Path
from typing import Optional

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

# LlamaParse
try:
    from llama_parse import LlamaParse
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False
    print("Warning: llama-parse not installed. Run: pip install llama-parse")

from src.config import settings
from src.indexing.extractors import get_all_extractors

# Vision fallback
try:
    from src.indexing.vision_extractor import (
        extract_text_with_vision,
        get_pdf_page_count,
        should_use_vision,
        PYMUPDF_AVAILABLE,
    )
    VISION_AVAILABLE = PYMUPDF_AVAILABLE
except ImportError:
    VISION_AVAILABLE = False

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_PERSIST_DIR = PROJECT_ROOT / settings.CHROMA_PERSIST_DIR
NODES_CACHE_DIR = PROJECT_ROOT / ".llamaindex_nodes"


def extract_metadata_from_filename(filename: str) -> dict:
    """
    ファイル名からベースメタデータを抽出

    Args:
        filename: ファイル名

    Returns:
        メタデータ辞書
    """
    metadata = {"filename": filename}

    # エリア抽出
    area_names = [
        "東京電力", "関西電力", "中部電力", "九州電力", "東北電力",
        "北海道電力", "中国電力", "四国電力", "北陸電力", "沖縄電力"
    ]
    for area in area_names:
        if area in filename:
            metadata["area"] = area
            break

    # 電圧タイプ抽出
    if "高圧特別高圧" in filename:
        metadata["voltage_type"] = "高圧特別高圧"
    elif "特別高圧" in filename:
        metadata["voltage_type"] = "特別高圧"
    elif "高圧" in filename and "低圧" not in filename:
        metadata["voltage_type"] = "高圧"
    elif "低圧" in filename:
        metadata["voltage_type"] = "低圧"

    # ドキュメントタイプ
    if "託送" in filename:
        metadata["doc_type"] = "託送供給等約款"
    elif "約款" in filename:
        metadata["doc_type"] = "約款"
    elif "マニュアル" in filename or "manual" in filename.lower():
        metadata["doc_type"] = "システムマニュアル"
    elif "qa" in filename.lower() or "Q&A" in filename:
        metadata["doc_type"] = "Q&A"
    else:
        metadata["doc_type"] = "その他"

    return metadata


def parse_pdf_with_llamaparse(pdf_path: Path) -> list[Document]:
    """
    LlamaParseでPDFをパースしてLlamaIndex Documentに変換

    PDF_VISION_ENABLEDが有効な場合、テキスト抽出が不十分なページは
    GPT-4o Visionで再抽出する（ページ単位のフォールバック）。

    Args:
        pdf_path: PDFファイルのパス

    Returns:
        LlamaIndex Documentのリスト
    """
    if not LLAMAPARSE_AVAILABLE:
        raise ImportError("llama-parse is not installed")

    if not settings.LLAMA_CLOUD_API_KEY:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY が設定されていません。\n"
            "https://cloud.llamaindex.ai/ でAPIキーを取得し、.envに設定してください。"
        )

    print(f"  LlamaParseで解析中: {pdf_path.name}")

    parser = LlamaParse(
        api_key=settings.LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        num_workers=2,
        verbose=False,
        language="ja",
    )

    # LlamaParseでパース
    parsed_docs = parser.load_data(str(pdf_path))

    if not parsed_docs:
        print(f"  警告: パース結果が空です: {pdf_path.name}")
        return []

    # LlamaIndex Documentに変換
    base_metadata = extract_metadata_from_filename(pdf_path.name)
    base_metadata["source"] = str(pdf_path)

    documents = []
    vision_pages = []

    for i, doc in enumerate(parsed_docs):
        page_num = i + 1
        text = doc.text
        parser_used = "llamaparse"

        # Vision フォールバック判定（ページ単位）
        if (
            settings.PDF_VISION_ENABLED
            and VISION_AVAILABLE
            and should_use_vision(text)
        ):
            try:
                print(f"    ページ{page_num}: テキスト不足({len(text)}文字) → Vision抽出中...")
                text = extract_text_with_vision(pdf_path, i)  # 0-indexed
                parser_used = "vision"
                vision_pages.append(page_num)
            except Exception as e:
                print(f"    ページ{page_num}: Vision抽出失敗 ({e}), LlamaParse結果を使用")

        metadata = {**base_metadata, "page": page_num, "parser": parser_used}
        documents.append(Document(text=text, metadata=metadata))

    # 完了メッセージ
    total_chars = sum(len(d.text) for d in documents)
    vision_info = f", Vision使用: {len(vision_pages)}ページ" if vision_pages else ""
    print(f"  完了: {len(documents)}ページ, {total_chars:,}文字{vision_info}")

    return documents


def load_text_file(file_path: Path) -> list[Document]:
    """テキストファイルを読み込み"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    metadata = extract_metadata_from_filename(file_path.name)
    metadata["source"] = str(file_path)
    metadata["parser"] = "text"

    return [Document(text=text, metadata=metadata)]


def load_csv_file(file_path: Path) -> list[Document]:
    """CSVファイルを読み込み（各行を個別のDocumentに）"""
    import csv

    documents = []
    base_metadata = extract_metadata_from_filename(file_path.name)
    base_metadata["source"] = str(file_path)
    base_metadata["parser"] = "csv"
    base_metadata["doc_type"] = "Q&A"

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # 行の内容を文字列に結合
            text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
            if text.strip():
                metadata = {**base_metadata, "row": i + 1}
                documents.append(Document(text=text, metadata=metadata))

    return documents


def create_ingestion_pipeline() -> IngestionPipeline:
    """
    IngestionPipelineを作成

    Returns:
        設定済みのIngestionPipeline
    """
    # SentenceSplitterで文脈を保持しつつ分割
    # chunk_size=1024: 約1000文字のチャンク（LangGraph版と同等）
    # chunk_overlap=200: オーバーラップで文脈を保持
    node_parser = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )

    # カスタム抽出器
    extractors = get_all_extractors()

    pipeline = IngestionPipeline(
        transformations=[
            node_parser,
            *extractors,
        ]
    )

    return pipeline


def add_context_prefix_to_nodes(nodes: list[TextNode]) -> list[TextNode]:
    """
    ノードにコンテキスト接頭辞を追加

    電圧タイプやエリア情報を検索しやすくするため、
    テキストの先頭にメタデータを付与する。
    """
    for node in nodes:
        metadata = node.metadata
        prefix_parts = []

        if metadata.get("area"):
            prefix_parts.append(f"【{metadata['area']}エリア】")
        if metadata.get("voltage_type"):
            prefix_parts.append(f"【{metadata['voltage_type']}】")
        if metadata.get("section_title"):
            prefix_parts.append(f"【{metadata['section_title'][:30]}】")

        if prefix_parts:
            prefix = " ".join(prefix_parts) + "\n"
            node.text = prefix + node.text

    return nodes


def create_llamaindex_index(force: bool = False, max_pdfs: Optional[int] = None):
    """
    LlamaIndexでインデックスを作成

    Args:
        force: 既存インデックスを削除して再作成
        max_pdfs: 処理するPDFの最大数（テスト用）
    """
    if CHROMA_PERSIST_DIR.exists():
        if force:
            print("既存のインデックスを削除します...")
            shutil.rmtree(CHROMA_PERSIST_DIR)
        else:
            print(f"インデックスは既に存在します: {CHROMA_PERSIST_DIR}")
            print("再作成する場合は --force オプションを付けてください。")
            return

    print("\n=== LlamaIndex インデックス作成 ===\n")

    all_documents = []

    # PDFファイルを処理
    print("=== PDFファイルの処理（LlamaParse）===")
    pdf_files = list(DOCUMENTS_DIR.glob("**/*.pdf"))
    if max_pdfs:
        pdf_files = pdf_files[:max_pdfs]

    for pdf_path in pdf_files:
        try:
            docs = parse_pdf_with_llamaparse(pdf_path)
            all_documents.extend(docs)
        except Exception as e:
            print(f"  エラー ({pdf_path.name}): {e}")

    # テキストファイルを処理
    print("\n=== テキストファイルの処理 ===")
    for txt_path in DOCUMENTS_DIR.glob("**/*.txt"):
        try:
            docs = load_text_file(txt_path)
            all_documents.extend(docs)
            print(f"  {txt_path.name}: {len(docs)}件")
        except Exception as e:
            print(f"  エラー ({txt_path.name}): {e}")

    # CSVファイルを処理
    print("\n=== CSVファイルの処理 ===")
    for csv_path in DOCUMENTS_DIR.glob("**/*.csv"):
        try:
            docs = load_csv_file(csv_path)
            all_documents.extend(docs)
            print(f"  {csv_path.name}: {len(docs)}件")
        except Exception as e:
            print(f"  エラー ({csv_path.name}): {e}")

    if not all_documents:
        print("処理するドキュメントがありません。")
        return

    print(f"\n総ドキュメント数: {len(all_documents)}")

    # IngestionPipelineでノードに変換
    print("\n=== IngestionPipelineでノード変換 ===")
    pipeline = create_ingestion_pipeline()
    nodes = pipeline.run(documents=all_documents, show_progress=True)
    print(f"ノード数: {len(nodes)}")

    # コンテキスト接頭辞を追加
    nodes = add_context_prefix_to_nodes(nodes)

    # ChromaDBベクトルストアを作成
    print("\n=== ベクトルストア作成 ===")
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    chroma_collection = chroma_client.get_or_create_collection("llamaindex_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Embeddingモデル
    embed_model = OpenAIEmbedding(model=settings.EMBEDDING_MODEL)

    # インデックス作成
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print(f"\nインデックスを保存しました: {CHROMA_PERSIST_DIR}")

    # 統計情報
    print("\n=== 統計情報 ===")
    stats = {"doc_type": {}, "has_table": 0, "with_voltage": 0, "with_area": 0}

    for node in nodes:
        metadata = node.metadata

        dt = metadata.get("doc_type", "不明")
        stats["doc_type"][dt] = stats["doc_type"].get(dt, 0) + 1

        if metadata.get("has_table"):
            stats["has_table"] += 1
        if metadata.get("voltage_type"):
            stats["with_voltage"] += 1
        if metadata.get("area"):
            stats["with_area"] += 1

    print("ドキュメントタイプ別:")
    for dt, count in sorted(stats["doc_type"].items(), key=lambda x: -x[1]):
        print(f"  {dt}: {count}件")

    print(f"\nテーブル含むノード: {stats['has_table']}件")
    print(f"電圧タイプあり: {stats['with_voltage']}件")
    print(f"エリアあり: {stats['with_area']}件")


def get_index() -> VectorStoreIndex:
    """
    保存済みのインデックスを取得

    Returns:
        VectorStoreIndex
    """
    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"インデックスが見つかりません: {CHROMA_PERSIST_DIR}\n"
            "先に 'python -m src.indexing.llamaindex_pipeline' を実行してください。"
        )

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    chroma_collection = chroma_client.get_collection("llamaindex_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = OpenAIEmbedding(model=settings.EMBEDDING_MODEL)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    return index


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    max_pdfs = None

    for arg in sys.argv[1:]:
        if arg.startswith("--max="):
            max_pdfs = int(arg.split("=")[1])

    create_llamaindex_index(force=force, max_pdfs=max_pdfs)
