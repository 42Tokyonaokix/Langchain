"""
インデックス作成スクリプト

ドキュメントを読み込んでChromaDBにインデックスを作成します。
このスクリプトは事前に1回だけ実行してください。
"""

import os
import re
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def extract_section_title(content):
    """
    コンテンツからセクションタイトルを抽出
    """
    # セクションタイトルのパターン
    patterns = [
        # システムマニュアル: "1. ログイン", "7. 請求書送付先アドレス追加・変更"
        r'^(\d{1,2}\.\s*[^\n]{2,50})$',
        # 約款の条項: "第1条（目的）", "第12条"
        r'^(第\d{1,3}条[^\n]{0,50})$',
        # 約款の項目: "(1) 〜", "(ア) 〜"
        r'^(\(\d{1,2}\)\s*[^\n]{2,50})$',
        r'^(\([ア-ン]\)\s*[^\n]{2,50})$',
        # 大見出し: "DGMについて", "〜について"
        r'^([^\n]{2,20}について)$',
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return match.group(1).strip()

    return None


def clean_document(doc):
    """
    ドキュメントのノイズを除去し、ヘッダー情報をメタデータに移動
    """
    content = doc.page_content
    metadata = doc.metadata.copy()

    # ヘッダー情報をメタデータとして抽出
    if "Confidential" in content:
        metadata["confidential"] = True

    if "DIGITAL GRID" in content or "デジタルグリッド" in content:
        metadata["company"] = "Digital Grid Corporation"

    # ドキュメントタイプを判定してメタデータに追加
    source = metadata.get("source", "")
    if "system_manual" in source:
        metadata["doc_type"] = "システムマニュアル"
    elif "約款" in source:
        metadata["doc_type"] = "約款"
    elif "配送約款" in source or "託送" in source:
        metadata["doc_type"] = "託送供給等約款"
    elif "qa_list" in source:
        metadata["doc_type"] = "Q&A"

    # 電力エリア（送配電事業者）を抽出
    area_names = [
        "東京電力", "関西電力", "中部電力", "九州電力", "東北電力",
        "北海道電力", "中国電力", "四国電力", "北陸電力", "沖縄電力"
    ]
    for area in area_names:
        if area in source:
            metadata["area"] = area
            break

    # 電圧タイプを抽出（高圧/低圧）
    if "高圧特別高圧" in source:
        metadata["voltage_type"] = "高圧特別高圧"
    elif "特別高圧" in source:
        metadata["voltage_type"] = "特別高圧"
    elif "高圧" in source and "低圧" not in source:
        metadata["voltage_type"] = "高圧"
    elif "低圧" in source:
        metadata["voltage_type"] = "低圧"

    # セクションタイトルを抽出してメタデータに追加
    section = extract_section_title(content)
    if section:
        metadata["section"] = section

    # ノイズ除去
    # 1. ページ番号（行頭の数字のみの行）
    content = re.sub(r'^\s*\d{1,3}\s*$', '', content, flags=re.MULTILINE)

    # 2. プレースホルダー（##### kWh, #### 等）
    content = re.sub(r'#{3,}\s*\w*', '', content)
    content = re.sub(r'@##\.##', '', content)

    # 3. ヘッダー・フッター
    content = re.sub(r'Confidential\s*', '', content)
    content = re.sub(r'©︎?\s*DIGITAL GRID Corporation\s*', '', content)
    content = re.sub(r'©\s*DIGITAL GRID Corporation\s*', '', content)

    # 4. 連続する空白行を1つに
    content = re.sub(r'\n{3,}', '\n\n', content)

    # 5. 行頭・行末の空白を整理
    content = '\n'.join(line.strip() for line in content.split('\n'))
    content = content.strip()

    doc.page_content = content
    doc.metadata = metadata

    return doc

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_PERSIST_DIR = PROJECT_ROOT / ".chroma_db"


def load_documents():
    """documentsフォルダからドキュメントを読み込む"""
    if not DOCUMENTS_DIR.exists():
        print(f"エラー: ドキュメントフォルダが見つかりません: {DOCUMENTS_DIR}")
        return []

    all_docs = []

    # テキストファイル
    print("テキストファイルを読み込み中...")
    try:
        txt_loader = DirectoryLoader(
            str(DOCUMENTS_DIR),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        txt_docs = txt_loader.load()
        all_docs.extend(txt_docs)
        print(f"  → {len(txt_docs)}件")
    except Exception as e:
        print(f"  → エラー: {e}")

    # PDFファイル
    print("PDFファイルを読み込み中...")
    try:
        pdf_loader = DirectoryLoader(
            str(DOCUMENTS_DIR),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        pdf_docs = pdf_loader.load()
        all_docs.extend(pdf_docs)
        print(f"  → {len(pdf_docs)}件")
    except Exception as e:
        print(f"  → エラー: {e}")

    # CSVファイル
    print("CSVファイルを読み込み中...")
    try:
        csv_loader = DirectoryLoader(
            str(DOCUMENTS_DIR),
            glob="**/*.csv",
            loader_cls=CSVLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        csv_docs = csv_loader.load()
        all_docs.extend(csv_docs)
        print(f"  → {len(csv_docs)}件")
    except Exception as e:
        print(f"  → エラー: {e}")

    return all_docs


def add_context_prefix(chunk):
    """
    チャンクの本文にコンテキスト情報を追加
    （ファイル名のみに含まれる情報を本文にも追加して検索精度を向上）
    """
    source = chunk.metadata.get("source", "")
    prefix_parts = []

    # エリア情報を追加
    area = chunk.metadata.get("area", "")
    if area:
        prefix_parts.append(f"【{area}エリア】")

    # 電圧タイプを追加
    voltage = chunk.metadata.get("voltage_type", "")
    if voltage:
        prefix_parts.append(f"【{voltage}】")

    # ドキュメントタイプを追加
    doc_type = chunk.metadata.get("doc_type", "")
    if doc_type:
        prefix_parts.append(f"【{doc_type}】")

    if prefix_parts:
        prefix = " ".join(prefix_parts) + "\n"
        chunk.page_content = prefix + chunk.page_content

    return chunk


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """ドキュメントをチャンクに分割し、セクション情報を伝播"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # 各チャンクにセクション情報を追加/更新
    # 同じソース・ページのチャンクで、セクションがないものには前のチャンクのセクションを継承
    current_section = {}  # source+page -> section のマッピング

    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", 0)
        key = f"{source}:{page}"

        # チャンク自体にセクションタイトルがあるか確認
        chunk_section = extract_section_title(chunk.page_content)

        if chunk_section:
            # 新しいセクションを発見
            chunk.metadata["section"] = chunk_section
            current_section[key] = chunk_section
        elif "section" not in chunk.metadata and key in current_section:
            # セクションがなく、同じページの前のチャンクにセクションがあれば継承
            chunk.metadata["section"] = current_section[key]

    # コンテキスト情報をチャンク本文に追加（検索精度向上のため）
    chunks = [add_context_prefix(chunk) for chunk in chunks]

    return chunks


def create_index(force=False):
    """インデックスを作成"""
    # 既存のインデックスがある場合
    if CHROMA_PERSIST_DIR.exists():
        if force:
            print("既存のインデックスを削除します...")
            shutil.rmtree(CHROMA_PERSIST_DIR)
        else:
            print(f"インデックスは既に存在します: {CHROMA_PERSIST_DIR}")
            print("再作成する場合は --force オプションを付けてください。")
            return

    # ドキュメント読み込み
    print("\n=== ドキュメント読み込み ===")
    documents = load_documents()
    print(f"\n合計: {len(documents)}件のドキュメント")

    if not documents:
        print("エラー: ドキュメントが見つかりません")
        return

    # ノイズ除去・メタデータ抽出
    print("\n=== ノイズ除去・メタデータ抽出 ===")
    cleaned_docs = []
    for doc in documents:
        cleaned = clean_document(doc)
        # 空になったドキュメントはスキップ
        if cleaned.page_content.strip():
            cleaned_docs.append(cleaned)
    print(f"{len(documents)}件 → {len(cleaned_docs)}件（空ドキュメント除去後）")

    # チャンク分割
    print("\n=== チャンク分割 ===")
    chunks = split_documents(cleaned_docs)
    print(f"{len(cleaned_docs)}件 → {len(chunks)}チャンク")

    # ベクトルストア作成
    print("\n=== ベクトルストア作成 ===")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    print(f"インデックスを保存しました: {CHROMA_PERSIST_DIR}")
    print(f"チャンク数: {len(chunks)}")


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    create_index(force=force)
