"""PDF 解析：项目自研版面解析链路。

流程沿用已验证的自研解析逻辑：
1. `style_headings` 用 PDF 物理信号探测标题层级；
2. `pymupdf4llm` 逐页渲染 markdown（仅作为文本/版面底座）；
3. `clean_parse` 做表格重建、标题注入、伪影清洗与目录页剔除；
4. 输出逐页 markdown（页码对齐），供结构化分块与引用溯源。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import pymupdf4llm

from app.core.config import settings
from app.ingestion.custom_parse.clean_parse import (
    clean_pages,
    detect_structure,
    slim_chunks,
)

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    markdown: str
    page_texts: list[str] = field(default_factory=list)
    pages: list[dict] = field(default_factory=list)
    page_count: int = 0
    parser: str = "custom"
    page_aligned: bool = False
    image_count: int = 0

    @property
    def stats(self) -> dict:
        return {
            "parser": self.parser,
            "page_count": self.page_count,
            "page_aligned": self.page_aligned,
            "image_count": self.image_count,
            "markdown_chars": len(self.markdown),
        }


def parse_pdf(
    pdf_path: str | Path,
    work_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> ParsedDocument:
    """按 `PDF_PARSER` 调用自研解析；兼容历史 `auto` 配置值。"""
    mode = settings.pdf_parser.lower()
    if mode not in {"custom", "auto"}:
        raise ValueError(
            f"不支持的 PDF_PARSER：{settings.pdf_parser}（当前仅支持 custom）"
        )
    return _parse_with_custom(Path(pdf_path))


def _parse_with_custom(pdf_path: Path) -> ParsedDocument:
    with pymupdf.open(str(pdf_path)) as doc:
        page_count = len(doc)
        titles, heading_index, chapter_map = detect_structure(doc)
        raw = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True,
            header=False,
            footer=False,
            show_progress=False,
        )
        records, stats = clean_pages(
            slim_chunks(raw), titles, heading_index, doc=doc, chapter_map=chapter_map
        )

    if not records:
        raise RuntimeError("PDF 未解析出有效页面（可能全是目录/空页）")

    pages = [
        {"page_number": int(record["page"]), "text": record["text"]}
        for record in records
    ]
    markdown = "\n\n".join(page["text"] for page in pages)
    logger.info("自研解析完成：%s", stats)
    return ParsedDocument(
        markdown=markdown,
        pages=pages,
        page_count=page_count,
        parser="custom",
        page_aligned=True,
    )
