"""
LlamaParse + Agentic Chunking インデックス作成

特徴:
- LlamaParseで表・図を含むPDFを高精度にMarkdown変換
- 既存のagentic chunkingロジックでセクション分割
- リッチなメタデータ付与
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# LlamaParse
try:
    from llama_parse import LlamaParse
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False
    print("Warning: llama-parse not installed. Run: pip install llama-parse")

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_PERSIST_DIR = PROJECT_ROOT / ".chroma_db_llamaparse"
BM25_INDEX_PATH = PROJECT_ROOT / ".bm25_index_llamaparse.pkl"

# LLM設定
STRUCTURE_MODEL = "gpt-4o-mini"


def extract_area_from_filename(filename: str) -> Optional[str]:
    """ファイル名からエリア名を抽出"""
    area_names = [
        "東京電力", "関西電力", "中部電力", "九州電力", "東北電力",
        "北海道電力", "中国電力", "四国電力", "北陸電力", "沖縄電力"
    ]
    for area in area_names:
        if area in filename:
            return area
    return None


def extract_voltage_type(text: str) -> Optional[str]:
    """テキストから電圧タイプを抽出"""
    if "高圧特別高圧" in text:
        return "高圧特別高圧"
    if "特別高圧" in text:
        return "特別高圧"
    if "高圧" in text and "低圧" not in text:
        return "高圧"
    if "低圧" in text:
        return "低圧"
    return None


def extract_doc_type(filename: str) -> str:
    """ファイル名からドキュメントタイプを判定"""
    if "託送" in filename:
        return "託送供給等約款"
    if "約款" in filename:
        return "約款"
    if "マニュアル" in filename or "manual" in filename.lower():
        return "システムマニュアル"
    if "qa" in filename.lower() or "Q&A" in filename:
        return "Q&A"
    return "その他"


def parse_markdown_sections(markdown_text: str) -> list[dict]:
    """
    LlamaParseが出力したMarkdownをセクションに分割

    Markdownの見出し（#, ##, ###）やテーブルを認識
    """
    sections = []

    # 見出しパターン（Markdown + 日本の法的文書）
    section_patterns = [
        r'^(#{1,3})\s+(.+)$',           # Markdown見出し
        r'^(第\d{1,3}条[（\(][^）\)]+[）\)]?)',  # 第X条（タイトル）
        r'^(\d{1,2})\s+([^\n]{2,30})$',  # 番号 タイトル
    ]

    lines = markdown_text.split('\n')
    current_section = {
        "title": "",
        "level": 0,
        "content": [],
        "has_table": False,
        "start_line": 0,
    }

    for i, line in enumerate(lines):
        # テーブル検出
        if '|' in line and '-|-' not in line:
            current_section["has_table"] = True

        # セクション見出し検出
        is_heading = False
        for pattern in section_patterns:
            match = re.match(pattern, line.strip())
            if match:
                # 前のセクションを保存
                if current_section["content"] or current_section["title"]:
                    sections.append(current_section.copy())

                # 新しいセクション開始
                if match.group(1).startswith('#'):
                    level = len(match.group(1))
                    title = match.group(2)
                else:
                    level = 1
                    title = line.strip()

                current_section = {
                    "title": title,
                    "level": level,
                    "content": [line],
                    "has_table": False,
                    "start_line": i,
                }
                is_heading = True
                break

        if not is_heading:
            current_section["content"].append(line)

    # 最後のセクションを保存
    if current_section["content"]:
        sections.append(current_section)

    return sections


def smart_chunk_markdown(
    markdown_text: str,
    base_metadata: dict,
    max_chunk_size: int = 800,
) -> list[Document]:
    """
    Markdownテキストをスマートにチャンク分割

    - 表はできるだけ分割しない
    - セクション境界を尊重
    - メタデータを付与
    """
    sections = parse_markdown_sections(markdown_text)
    documents = []

    for section in sections:
        content = '\n'.join(section["content"]).strip()

        if not content:
            continue

        # セクションメタデータ
        section_metadata = {
            **base_metadata,
            "section_title": section["title"],
            "has_table": section["has_table"],
        }

        # コンテキスト接頭辞を追加
        prefix_parts = []
        if base_metadata.get("area"):
            prefix_parts.append(f"【{base_metadata['area']}エリア】")
        if base_metadata.get("voltage_type"):
            prefix_parts.append(f"【{base_metadata['voltage_type']}】")
        if section["title"]:
            prefix_parts.append(f"【{section['title'][:30]}】")

        prefix = " ".join(prefix_parts) + "\n" if prefix_parts else ""

        # サイズチェック
        if len(content) <= max_chunk_size:
            documents.append(Document(
                page_content=prefix + content,
                metadata=section_metadata,
            ))
        else:
            # 大きいセクションは分割
            # テーブルがある場合はテーブル単位で分割を試みる
            if section["has_table"]:
                chunks = split_preserving_tables(content, max_chunk_size)
            else:
                chunks = split_by_paragraphs(content, max_chunk_size)

            for chunk in chunks:
                if chunk.strip():
                    documents.append(Document(
                        page_content=prefix + chunk,
                        metadata=section_metadata.copy(),
                    ))

    return documents


def split_preserving_tables(content: str, max_size: int) -> list[str]:
    """テーブルを保持しながら分割"""
    chunks = []

    # テーブルブロックを検出（|で始まる連続行）
    table_pattern = r'(\|[^\n]+\|\n?)+'

    # テーブルとテキストを分離
    parts = re.split(f'({table_pattern})', content)

    current_chunk = ""
    for part in parts:
        if not part.strip():
            continue

        # テーブルかどうか判定
        is_table = part.strip().startswith('|')

        if is_table:
            # テーブルは分割しない（max_sizeを超えても1つのチャンクに）
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.append(part)
        else:
            if len(current_chunk) + len(part) <= max_size:
                current_chunk += part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def split_by_paragraphs(content: str, max_size: int) -> list[str]:
    """段落単位で分割"""
    paragraphs = content.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_size:
            current_chunk += ("\n\n" + para if current_chunk else para)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def process_pdf_with_llamaparse(pdf_path: Path) -> list[Document]:
    """
    LlamaParseでPDFを処理
    """
    print(f"\n処理中: {pdf_path.name}")

    if not LLAMAPARSE_AVAILABLE:
        raise ImportError("llama-parse is not installed")

    # LlamaParse APIキー確認
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY が設定されていません。\n"
            "https://cloud.llamaindex.ai/ でAPIキーを取得し、.envに設定してください。"
        )

    # LlamaParseで解析
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",  # Markdown形式で出力（表を保持）
        num_workers=2,
        verbose=True,
        language="ja",  # 日本語
    )

    print("  LlamaParseで解析中...")
    parsed_docs = parser.load_data(str(pdf_path))

    if not parsed_docs:
        print("  警告: パース結果が空です")
        return []

    # 全ページのMarkdownを結合
    full_markdown = "\n\n".join([doc.text for doc in parsed_docs])
    print(f"  Markdown変換完了: {len(full_markdown):,}文字")

    # ファイル名からメタデータを抽出
    filename = pdf_path.name
    base_metadata = {
        "source": str(pdf_path),
        "filename": filename,
        "parser": "llamaparse",
        "area": extract_area_from_filename(filename),
        "voltage_type": extract_voltage_type(filename),
        "doc_type": extract_doc_type(filename),
    }
    # Noneを除去
    base_metadata = {k: v for k, v in base_metadata.items() if v is not None}

    # Markdownをスマートにチャンク分割
    documents = smart_chunk_markdown(full_markdown, base_metadata)
    print(f"  チャンク数: {len(documents)}")

    return documents


def process_with_fallback(pdf_path: Path) -> list[Document]:
    """
    LlamaParseで失敗した場合はPyPDFLoaderにフォールバック
    """
    try:
        return process_pdf_with_llamaparse(pdf_path)
    except Exception as e:
        print(f"  LlamaParse失敗: {e}")
        print("  PyPDFLoaderにフォールバック...")

        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )

        filename = pdf_path.name
        for page in pages:
            page.metadata.update({
                "source": str(pdf_path),
                "filename": filename,
                "parser": "pypdf_fallback",
                "area": extract_area_from_filename(filename),
                "voltage_type": extract_voltage_type(filename),
                "doc_type": extract_doc_type(filename),
            })

        return splitter.split_documents(pages)


def create_llamaparse_index(force: bool = False, max_pdfs: int = None):
    """
    LlamaParse + Agentic Chunkingでインデックスを作成
    """
    if CHROMA_PERSIST_DIR.exists():
        if force:
            print("既存のインデックスを削除します...")
            shutil.rmtree(CHROMA_PERSIST_DIR)
            if BM25_INDEX_PATH.exists():
                BM25_INDEX_PATH.unlink()
        else:
            print(f"インデックスは既に存在します: {CHROMA_PERSIST_DIR}")
            print("再作成する場合は --force オプションを付けてください。")
            return

    all_documents = []

    # PDFファイルを処理
    print("\n=== PDFファイルの処理（LlamaParse）===")
    pdf_files = list(DOCUMENTS_DIR.glob("**/*.pdf"))
    if max_pdfs:
        pdf_files = pdf_files[:max_pdfs]

    for pdf_path in pdf_files:
        try:
            docs = process_with_fallback(pdf_path)
            all_documents.extend(docs)
        except Exception as e:
            print(f"  エラー: {e}")

    # テキスト・CSVファイルも処理（既存ロジック）
    print("\n=== テキスト/CSVファイルの処理 ===")
    from langchain_community.document_loaders import TextLoader, CSVLoader

    for txt_path in DOCUMENTS_DIR.glob("**/*.txt"):
        try:
            loader = TextLoader(str(txt_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(txt_path)
                doc.metadata["doc_type"] = "テキスト"
            all_documents.extend(docs)
            print(f"  {txt_path.name}: {len(docs)}件")
        except Exception as e:
            print(f"  エラー ({txt_path.name}): {e}")

    for csv_path in DOCUMENTS_DIR.glob("**/*.csv"):
        try:
            loader = CSVLoader(str(csv_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(csv_path)
                doc.metadata["doc_type"] = "Q&A"
            all_documents.extend(docs)
            print(f"  {csv_path.name}: {len(docs)}件")
        except Exception as e:
            print(f"  エラー ({csv_path.name}): {e}")

    # 空ドキュメントを除去
    all_documents = [doc for doc in all_documents if doc.page_content.strip()]

    print(f"\n=== ベクトルストア作成 ===")
    print(f"総チャンク数: {len(all_documents)}")

    # ベクトルストア作成
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    print(f"インデックスを保存しました: {CHROMA_PERSIST_DIR}")

    # 統計情報
    print("\n=== 統計情報 ===")
    stats = {"doc_type": {}, "parser": {}, "has_table": 0}

    for doc in all_documents:
        dt = doc.metadata.get("doc_type", "不明")
        stats["doc_type"][dt] = stats["doc_type"].get(dt, 0) + 1

        parser = doc.metadata.get("parser", "other")
        stats["parser"][parser] = stats["parser"].get(parser, 0) + 1

        if doc.metadata.get("has_table"):
            stats["has_table"] += 1

    print("ドキュメントタイプ別:")
    for dt, count in sorted(stats["doc_type"].items(), key=lambda x: -x[1]):
        print(f"  {dt}: {count}件")

    print("パーサー別:")
    for parser, count in sorted(stats["parser"].items(), key=lambda x: -x[1]):
        print(f"  {parser}: {count}件")

    print(f"テーブル含むチャンク: {stats['has_table']}件")


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    max_pdfs = None

    for arg in sys.argv[1:]:
        if arg.startswith("--max="):
            max_pdfs = int(arg.split("=")[1])

    create_llamaparse_index(force=force, max_pdfs=max_pdfs)
