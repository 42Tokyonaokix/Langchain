"""
LlamaIndex カスタムメタデータ抽出器

電力系ドキュメント向けのメタデータ抽出を行う。
"""

import re
from typing import ClassVar, Optional, Sequence

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.extractors import BaseExtractor


class VoltageTypeExtractor(BaseExtractor):
    """
    電圧タイプを抽出するExtractor

    高圧、低圧、特別高圧、高圧特別高圧を検出してメタデータに付与
    """

    def __init__(self):
        super().__init__()

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        """ノードから電圧タイプを抽出"""
        results = []
        for node in nodes:
            text = node.get_content()
            metadata = node.metadata if hasattr(node, 'metadata') else {}

            # ファイル名とコンテンツ両方をチェック
            source_text = metadata.get("filename", "") + " " + text

            voltage_type = self._detect_voltage_type(source_text)

            if voltage_type:
                results.append({"voltage_type": voltage_type})
            else:
                results.append({})

        return results

    def _detect_voltage_type(self, text: str) -> Optional[str]:
        """テキストから電圧タイプを検出"""
        if "高圧特別高圧" in text or ("高圧" in text and "特別高圧" in text):
            return "高圧特別高圧"
        if "特別高圧" in text:
            return "特別高圧"
        if "高圧" in text and "低圧" not in text:
            return "高圧"
        if "低圧" in text:
            return "低圧"
        return None


class AreaExtractor(BaseExtractor):
    """
    エリア（電力会社）を抽出するExtractor
    """

    AREA_NAMES: ClassVar[list[str]] = [
        "東京電力", "関西電力", "中部電力", "九州電力", "東北電力",
        "北海道電力", "中国電力", "四国電力", "北陸電力", "沖縄電力"
    ]

    def __init__(self):
        super().__init__()

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        """ノードからエリア情報を抽出"""
        results = []
        for node in nodes:
            text = node.get_content()
            metadata = node.metadata if hasattr(node, 'metadata') else {}

            # ファイル名とコンテンツ両方をチェック
            source_text = metadata.get("filename", "") + " " + text

            area = self._detect_area(source_text)

            if area:
                results.append({"area": area})
            else:
                results.append({})

        return results

    def _detect_area(self, text: str) -> Optional[str]:
        """テキストからエリアを検出"""
        for area in self.AREA_NAMES:
            if area in text:
                return area
        return None


class DocumentTypeExtractor(BaseExtractor):
    """
    ドキュメント種別を抽出するExtractor
    """

    def __init__(self):
        super().__init__()

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        """ノードからドキュメント種別を抽出"""
        results = []
        for node in nodes:
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            filename = metadata.get("filename", "")

            doc_type = self._detect_doc_type(filename)
            results.append({"doc_type": doc_type})

        return results

    def _detect_doc_type(self, filename: str) -> str:
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


class TableContextExtractor(BaseExtractor):
    """
    表の文脈情報を付与するExtractor

    表を含むノードに対して、親セクションのタイトルや文脈を付与し、
    表だけでは不足する検索可能性を補う。
    """

    context_window: int = 200

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        """ノードの表コンテキストを抽出"""
        results = []
        for node in nodes:
            text = node.get_content()
            metadata = node.metadata if hasattr(node, 'metadata') else {}

            has_table = self._has_table(text)
            result = {"has_table": has_table}

            if has_table:
                # 表の前後のテキストをコンテキストとして抽出
                table_context = self._extract_table_context(text)
                if table_context:
                    result["table_context"] = table_context

            results.append(result)

        return results

    def _has_table(self, text: str) -> bool:
        """テキストにMarkdownテーブルが含まれているか判定"""
        # Markdownテーブルパターン: |で区切られた行
        table_pattern = r'\|[^\n]+\|'
        separator_pattern = r'\|[-:]+\|'

        has_rows = bool(re.search(table_pattern, text))
        has_separator = bool(re.search(separator_pattern, text))

        return has_rows and has_separator

    def _extract_table_context(self, text: str) -> Optional[str]:
        """
        表の前後からコンテキストを抽出

        表の直前の見出しや説明文を取得して、表の内容を補足する
        """
        lines = text.split('\n')
        context_parts = []

        in_table = False
        pre_table_lines = []

        for line in lines:
            is_table_line = '|' in line

            if is_table_line and not in_table:
                # テーブル開始前の行をコンテキストとして保存
                in_table = True
                # 直前の非空行を最大3行取得
                non_empty = [l for l in pre_table_lines if l.strip()][-3:]
                context_parts.extend(non_empty)

            if not is_table_line:
                in_table = False
                pre_table_lines.append(line)
                # バッファは最新10行のみ保持
                pre_table_lines = pre_table_lines[-10:]

        if context_parts:
            return '\n'.join(context_parts)[:self.context_window]
        return None


class SectionTitleExtractor(BaseExtractor):
    """
    セクションタイトルを抽出するExtractor

    Markdownの見出しや法的文書の条文番号を検出
    """

    def __init__(self):
        super().__init__()

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        """ノードからセクションタイトルを抽出"""
        results = []
        for node in nodes:
            text = node.get_content()

            section_title = self._extract_section_title(text)

            if section_title:
                results.append({"section_title": section_title})
            else:
                results.append({})

        return results

    def _extract_section_title(self, text: str) -> Optional[str]:
        """テキストからセクションタイトルを抽出"""
        lines = text.split('\n')

        for line in lines[:5]:  # 最初の5行をチェック
            line = line.strip()

            # Markdown見出し
            match = re.match(r'^#{1,3}\s+(.+)$', line)
            if match:
                return match.group(1)[:50]

            # 第X条形式
            match = re.match(r'^(第\d{1,3}条[（\(][^）\)]+[）\)]?)', line)
            if match:
                return match.group(1)[:50]

            # 【タイトル】形式
            match = re.match(r'^【([^】]+)】', line)
            if match:
                return match.group(1)[:50]

        return None


def get_all_extractors() -> list[BaseExtractor]:
    """
    全ての抽出器を取得

    Returns:
        LlamaIndex IngestionPipelineで使用する抽出器のリスト
    """
    return [
        VoltageTypeExtractor(),
        AreaExtractor(),
        DocumentTypeExtractor(),
        TableContextExtractor(),
        SectionTitleExtractor(),
    ]
