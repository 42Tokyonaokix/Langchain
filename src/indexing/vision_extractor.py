"""
Vision PDF Extractor

LlamaParseでテキスト抽出が不十分なページをGPT-4o Visionで再抽出する。
ページ単位で判定し、必要なページだけVisionを使用してコストを最適化。
"""

import base64
from pathlib import Path
from io import BytesIO

from openai import OpenAI

from src.config import settings

# PDF to image conversion
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def pdf_page_to_base64(pdf_path: Path, page_num: int, dpi: int = 150) -> str:
    """
    PDFの指定ページを画像化してbase64エンコード

    Args:
        pdf_path: PDFファイルのパス
        page_num: ページ番号（0始まり）
        dpi: 解像度（デフォルト150）

    Returns:
        base64エンコードされた画像データ
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError(
            "PyMuPDFがインストールされていません。\n"
            "pip install pymupdf を実行してください。"
        )

    doc = fitz.open(str(pdf_path))
    page = doc.load_page(page_num)

    # ページを画像化
    zoom = dpi / 72  # 72dpiがデフォルト
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)

    # PNG形式でbase64エンコード
    img_bytes = pix.tobytes("png")
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    doc.close()

    return base64_image


def extract_text_with_vision(
    pdf_path: Path,
    page_num: int,
    dpi: int = 150,
) -> str:
    """
    GPT-4o Visionを使ってPDFページからテキストを抽出

    Args:
        pdf_path: PDFファイルのパス
        page_num: ページ番号（0始まり）
        dpi: 画像解像度

    Returns:
        抽出されたテキスト
    """
    # ページを画像化
    base64_image = pdf_page_to_base64(pdf_path, page_num, dpi)

    # OpenAI Visionで抽出
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=settings.PDF_VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたはPDFドキュメントからテキストを抽出する専門家です。"
                    "画像に含まれるすべてのテキストを正確に抽出してください。"
                    "表がある場合はMarkdown形式で再現してください。"
                    "図やグラフがある場合は、その内容を説明してください。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "この画像からすべてのテキストを抽出してください。表はMarkdown形式で、図は説明文で出力してください。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=4096,
        temperature=0,
    )

    return response.choices[0].message.content or ""


def get_pdf_page_count(pdf_path: Path) -> int:
    """PDFのページ数を取得"""
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDFがインストールされていません。")

    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def should_use_vision(text: str, threshold: int | None = None) -> bool:
    """
    Visionを使うべきかどうかを判定

    Args:
        text: LlamaParseで抽出されたテキスト
        threshold: 文字数閾値（Noneの場合は設定値を使用）

    Returns:
        True: Visionを使うべき, False: LlamaParseの結果で十分
    """
    if threshold is None:
        threshold = settings.PDF_VISION_FALLBACK_THRESHOLD

    # 空白を除いた実質的な文字数で判定
    actual_chars = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))

    return actual_chars < threshold


__all__ = [
    "extract_text_with_vision",
    "pdf_page_to_base64",
    "get_pdf_page_count",
    "should_use_vision",
    "PYMUPDF_AVAILABLE",
]
