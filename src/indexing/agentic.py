"""
Agentic Document Splitter - 最強のインデックス作成

特徴:
- LLMを使って文書構造を解析
- 章・節ごとにインテリジェントに分割
- 表データも適切に処理
- 各チャンクにリッチなメタデータを付与
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# パス設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_PERSIST_DIR = PROJECT_ROOT / ".chroma_db_agentic"

# LLM設定
STRUCTURE_MODEL = "gpt-4o-mini"  # 構造解析用（コスト効率）
CHUNK_MODEL = "gpt-4o-mini"      # チャンク分割用


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


def extract_document_structure(pages: list[Document], llm: ChatOpenAI) -> dict:
    """
    LLMを使って文書構造を解析
    目次がある場合は目次から、ない場合は全体から構造を推定
    """
    # 最初の10ページを確認（目次を探す）
    first_pages = "\n".join([p.page_content for p in pages[:10]])

    prompt = f"""以下の文書の最初の部分を読み、文書の構造（章・節）を抽出してください。

【文書の冒頭】
{first_pages[:8000]}

【出力形式】
JSON形式で出力してください:
{{
    "document_type": "文書の種類（約款、マニュアルなど）",
    "sections": [
        {{"number": "1", "title": "セクションタイトル", "keywords": ["キーワード1", "キーワード2"]}},
        ...
    ]
}}

セクションは主要な章・節のみを抽出し、50件以内にしてください。
"""

    response = llm.invoke(prompt)

    # JSONを抽出
    try:
        # コードブロックを除去
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content)
    except:
        return {"document_type": "unknown", "sections": []}


def find_section_boundaries(pages: list[Document], sections: list[dict]) -> list[dict]:
    """
    各セクションの開始ページと位置を特定
    """
    full_text = "\n".join([f"[PAGE {i+1}]\n{p.page_content}" for i, p in enumerate(pages)])

    boundaries = []
    for section in sections:
        title = section.get("title", "")
        number = section.get("number", "")

        # セクションタイトルのパターンを検索（複数パターン）
        patterns = []

        # タイトルがある場合
        if title:
            title_escaped = re.escape(title[:15])
            patterns.extend([
                rf"{re.escape(number)}\s*{title_escaped}",
                rf"第\s*{re.escape(number)}\s*条",
                rf"{re.escape(number)}\s+[^\n]{{0,5}}{title_escaped[:8]}",
            ])

        # 数字のみのパターン
        if number:
            patterns.extend([
                rf"\n{re.escape(number)}\s+\S",
                rf"第{re.escape(number)}条",
                rf"Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ" if number in ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "1", "2", "3", "4", "5"] else None,
            ])

        patterns = [p for p in patterns if p]

        for pattern in patterns:
            try:
                match = re.search(pattern, full_text)
                if match:
                    # ページ番号を抽出
                    before_match = full_text[:match.start()]
                    page_matches = list(re.finditer(r'\[PAGE (\d+)\]', before_match))
                    if page_matches:
                        page_num = int(page_matches[-1].group(1))
                    else:
                        page_num = 1

                    boundaries.append({
                        **section,
                        "start_page": page_num,
                        "start_pos": match.start()
                    })
                    break
            except:
                continue

    return boundaries


def extract_sections_from_text(full_text: str, base_metadata: dict) -> list[dict]:
    """
    テキストから直接セクションを抽出（LLM不要のフォールバック）
    日本の約款の標準的な構造パターンを使用
    """
    sections = []

    # 条項パターン: 第1条、第12条など
    article_pattern = r'(第\d{1,3}条[（\(][^）\)]+[）\)]?)'
    for match in re.finditer(article_pattern, full_text):
        sections.append({
            "title": match.group(1),
            "start_pos": match.start(),
            "metadata": {**base_metadata, "section_title": match.group(1)}
        })

    # 大項目パターン: 1 適用、32 損失率など
    major_pattern = r'\n(\d{1,2})\s+([^\n]{2,20})\n'
    for match in re.finditer(major_pattern, full_text):
        sections.append({
            "title": f"{match.group(1)} {match.group(2)}",
            "start_pos": match.start(),
            "metadata": {**base_metadata, "section_number": match.group(1), "section_title": match.group(2)}
        })

    # 位置でソート
    sections.sort(key=lambda x: x["start_pos"])

    return sections


def smart_chunk_section(
    content: str,
    section_info: dict,
    llm: ChatOpenAI,
    max_chunk_size: int = 800,
    min_chunk_size: int = 100
) -> list[dict]:
    """
    セクションを意味的に分割
    表データや数値リストは保持しつつ、長いセクションは適切に分割
    """
    # 短いセクションはそのまま
    if len(content) <= max_chunk_size:
        return [{
            "content": content,
            "metadata": section_info
        }]

    # 長すぎる場合はまず条項・項目で分割を試みる
    chunks = []

    # 日本の法的文書の構造パターン
    # 第X条、(1)、イ、(イ)、ａなどで分割
    split_patterns = [
        r'(?=第\d{1,3}条)',           # 第1条、第12条など
        r'(?=\n\d{1,2}\s+[^\d\n])',    # "1 適用" など
        r'(?=\n\(\d+\)\s*)',           # (1)、(2)など
        r'(?=\n[イロハニホヘト]\s+)',  # イ、ロ、ハなど
    ]

    current_text = content
    for pattern in split_patterns:
        parts = re.split(pattern, current_text)
        if len(parts) > 1 and all(len(p) <= max_chunk_size * 2 for p in parts if p.strip()):
            current_text = "|||SPLIT|||".join(parts)
            break

    if "|||SPLIT|||" in current_text:
        parts = current_text.split("|||SPLIT|||")
    else:
        # パターンで分割できない場合は段落で分割
        parts = content.split("\n\n")

    # パーツを適切なサイズにまとめる
    current_chunk = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(current_chunk) + len(part) <= max_chunk_size:
            current_chunk += "\n\n" + part if current_chunk else part
        else:
            if current_chunk:
                chunks.append({
                    "content": current_chunk,
                    "metadata": section_info.copy()
                })
            current_chunk = part

    if current_chunk:
        chunks.append({
            "content": current_chunk,
            "metadata": section_info.copy()
        })

    # それでも大きすぎるチャンクがある場合はさらに分割
    final_chunks = []
    for chunk in chunks:
        if len(chunk["content"]) > max_chunk_size * 1.5:
            # 強制的に文単位で分割
            sentences = re.split(r'(?<=[。．\n])', chunk["content"])
            sub_chunk = ""
            for sent in sentences:
                if len(sub_chunk) + len(sent) <= max_chunk_size:
                    sub_chunk += sent
                else:
                    if sub_chunk:
                        final_chunks.append({
                            "content": sub_chunk,
                            "metadata": chunk["metadata"].copy()
                        })
                    sub_chunk = sent
            if sub_chunk:
                final_chunks.append({
                    "content": sub_chunk,
                    "metadata": chunk["metadata"].copy()
                })
        else:
            final_chunks.append(chunk)

    return final_chunks if final_chunks else [{"content": content, "metadata": section_info}]


def process_pdf_agentic(pdf_path: Path, llm: ChatOpenAI) -> list[Document]:
    """
    PDFをAgenticに処理
    """
    print(f"\n処理中: {pdf_path.name}")

    # PDFを読み込み
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    print(f"  ページ数: {len(pages)}")

    # ファイル名からメタデータを抽出
    filename = pdf_path.name
    area = extract_area_from_filename(filename)
    voltage = extract_voltage_type(filename)

    base_metadata = {
        "source": str(pdf_path),
        "filename": filename,
    }
    if area:
        base_metadata["area"] = area
    if voltage:
        base_metadata["voltage_type"] = voltage

    # ドキュメントタイプを判定
    if "託送" in filename:
        base_metadata["doc_type"] = "託送供給等約款"
    elif "約款" in filename:
        base_metadata["doc_type"] = "約款"
    elif "マニュアル" in filename or "manual" in filename.lower():
        base_metadata["doc_type"] = "システムマニュアル"

    # 文書構造を解析
    print("  構造解析中...")
    structure = extract_document_structure(pages, llm)
    print(f"  文書タイプ: {structure.get('document_type', 'unknown')}")
    print(f"  セクション数: {len(structure.get('sections', []))}")

    # セクション境界を特定
    sections = structure.get("sections", [])
    if sections:
        boundaries = find_section_boundaries(pages, sections)
        print(f"  境界検出: {len(boundaries)}セクション")
    else:
        boundaries = []

    # 全文を結合
    full_text = "\n".join([p.page_content for p in pages])

    # チャンクを作成
    documents = []

    # LLMベースの境界が少ない場合、パターンベースで補完
    if len(boundaries) < 5:
        print("  パターンベースのセクション抽出を追加...")
        pattern_sections = extract_sections_from_text(full_text, base_metadata)
        print(f"  パターン検出: {len(pattern_sections)}セクション")

        # 既存の境界と統合（重複除去）
        existing_positions = {b.get("start_pos", 0) for b in boundaries}
        for ps in pattern_sections:
            if ps["start_pos"] not in existing_positions:
                boundaries.append({
                    "title": ps["title"],
                    "start_pos": ps["start_pos"],
                    **ps.get("metadata", {})
                })

        # 位置でソート
        boundaries.sort(key=lambda x: x.get("start_pos", 0))
        print(f"  統合後: {len(boundaries)}セクション")

    if boundaries:
        # セクションごとに分割
        for i, boundary in enumerate(boundaries):
            # セクションの範囲を決定
            start_pos = boundary.get("start_pos", 0)
            if i + 1 < len(boundaries):
                end_pos = boundaries[i + 1].get("start_pos", len(full_text))
            else:
                end_pos = len(full_text)

            section_content = full_text[start_pos:end_pos].strip()

            if not section_content or len(section_content) < 50:
                continue

            # セクション情報
            section_info = {
                **base_metadata,
                "section_number": boundary.get("section_number", boundary.get("number", "")),
                "section_title": boundary.get("section_title", boundary.get("title", "")),
            }

            # セクションをスマートに分割
            chunks = smart_chunk_section(section_content, section_info, llm)

            for chunk in chunks:
                # コンテキスト接頭辞を追加
                prefix_parts = []
                if area:
                    prefix_parts.append(f"【{area}エリア】")
                if voltage:
                    prefix_parts.append(f"【{voltage}】")
                section_title = boundary.get("section_title", boundary.get("title", ""))
                if section_title:
                    prefix_parts.append(f"【{section_title[:30]}】")

                prefix = " ".join(prefix_parts) + "\n" if prefix_parts else ""

                doc = Document(
                    page_content=prefix + chunk["content"],
                    metadata=chunk["metadata"]
                )
                documents.append(doc)
    else:
        # セクション境界が全く検出できない場合はシンプル分割
        print("  セクション境界が検出できません。シンプル分割を使用します。")
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )

        for page in pages:
            page.metadata.update(base_metadata)

        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            # コンテキスト接頭辞を追加
            prefix_parts = []
            if area:
                prefix_parts.append(f"【{area}エリア】")
            if voltage:
                prefix_parts.append(f"【{voltage}】")

            prefix = " ".join(prefix_parts) + "\n" if prefix_parts else ""
            chunk.page_content = prefix + chunk.page_content
            documents.append(chunk)

    print(f"  チャンク数: {len(documents)}")
    return documents


def process_text_file(txt_path: Path) -> list[Document]:
    """テキストファイルを処理"""
    loader = TextLoader(str(txt_path), encoding="utf-8")
    docs = loader.load()

    # メタデータを追加
    for doc in docs:
        doc.metadata["source"] = str(txt_path)
        doc.metadata["doc_type"] = "テキスト"

    return docs


def process_csv_file(csv_path: Path) -> list[Document]:
    """CSVファイルを処理"""
    loader = CSVLoader(str(csv_path), encoding="utf-8")
    docs = loader.load()

    # メタデータを追加
    for doc in docs:
        doc.metadata["source"] = str(csv_path)
        doc.metadata["doc_type"] = "Q&A"

    return docs


def create_agentic_index(force: bool = False, max_pdfs: int = None):
    """
    Agenticインデックスを作成
    """
    # 既存のインデックスがある場合
    if CHROMA_PERSIST_DIR.exists():
        if force:
            print("既存のインデックスを削除します...")
            shutil.rmtree(CHROMA_PERSIST_DIR)
        else:
            print(f"インデックスは既に存在します: {CHROMA_PERSIST_DIR}")
            print("再作成する場合は --force オプションを付けてください。")
            return

    # LLM初期化
    llm = ChatOpenAI(model=STRUCTURE_MODEL, temperature=0)

    all_documents = []

    # PDFファイルを処理
    print("\n=== PDFファイルの処理 ===")
    pdf_files = list(DOCUMENTS_DIR.glob("**/*.pdf"))
    if max_pdfs:
        pdf_files = pdf_files[:max_pdfs]

    for pdf_path in pdf_files:
        try:
            docs = process_pdf_agentic(pdf_path, llm)
            all_documents.extend(docs)
        except Exception as e:
            print(f"  エラー: {e}")

    # テキストファイルを処理
    print("\n=== テキストファイルの処理 ===")
    for txt_path in DOCUMENTS_DIR.glob("**/*.txt"):
        try:
            docs = process_text_file(txt_path)
            all_documents.extend(docs)
            print(f"  {txt_path.name}: {len(docs)}件")
        except Exception as e:
            print(f"  エラー: {e}")

    # CSVファイルを処理
    print("\n=== CSVファイルの処理 ===")
    for csv_path in DOCUMENTS_DIR.glob("**/*.csv"):
        try:
            docs = process_csv_file(csv_path)
            all_documents.extend(docs)
            print(f"  {csv_path.name}: {len(docs)}件")
        except Exception as e:
            print(f"  エラー: {e}")

    # 空のドキュメントを除去
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
    doc_types = {}
    areas = {}
    for doc in all_documents:
        dt = doc.metadata.get("doc_type", "不明")
        doc_types[dt] = doc_types.get(dt, 0) + 1

        area = doc.metadata.get("area", "")
        if area:
            areas[area] = areas.get(area, 0) + 1

    print("ドキュメントタイプ別:")
    for dt, count in sorted(doc_types.items(), key=lambda x: -x[1]):
        print(f"  {dt}: {count}件")

    print("エリア別:")
    for area, count in sorted(areas.items(), key=lambda x: -x[1]):
        print(f"  {area}: {count}件")


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    max_pdfs = None

    for arg in sys.argv[1:]:
        if arg.startswith("--max="):
            max_pdfs = int(arg.split("=")[1])

    create_agentic_index(force=force, max_pdfs=max_pdfs)
