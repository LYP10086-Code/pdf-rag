# -*- coding: utf-8 -*-
"""提示框几何检测：标题条填充矩形 + 体部边框线（纯物理信号，零文本猜测）。

检测原理（秦PLUS EV 手册模板实测，全书 302 页零反例）：

1. 标题条：单 `re` 项的填充路径，宽高在配置范围内，填充色 ≈ BOX_BAR_FILL_HEX。
   条内文本匹配 警告/注意/温馨提示（含"（续）"）→ 确认为提示框（锚点校验）；
   同色但文本不匹配的小节标题条（如"安全气囊可能会展开的情况"）被排除。
2. 体部底边：标题条下方第一条与标题条 x 对齐的水平描边线（线宽≈1pt、深色），
   即框的精确结束坐标；标题条自带的底边线用 y > 条底+1 排除。
   找不到底边 = 跨页框首页，体部延伸到页面正文区底部。
3. 归属：文本行 bbox 中心 ∈ [标题条底, 底边] × 条 x 范围 → 框内内容。

输出按 y 升序的行文本清单（含标题行），供 clean_parse.mark_box_lines 对齐
markdown 行后包 ">" 引用块——分块层据此确定性切出纯净的提示框补充块。
"""

import re
from dataclasses import dataclass, field

import pymupdf

from app.core.config import settings
from app.ingestion.custom_parse.style_headings import norm_text

# 框标题（与 chunker.BOX_TITLE_LINE_RE 同词表；此处允许图标字符前缀）
TITLE_RE = re.compile(r"^[⚠!！]?\s*(警告|注意|温馨提示)\s*(?:[(（]\s*续\s*[)）])?\s*$")

# 体部边框线的深色校验（#231f20 实测值；排除橙色装饰线）
_DARK_CHANNEL_MAX = 0.35


@dataclass
class BoxRegion:
    page_number: int
    title: str
    lines: list[str] = field(default_factory=list)  # 标题行 + 内容行，按 y 升序


def _hex_to_rgb01(hex_str: str) -> tuple[float, float, float]:
    value = hex_str.lstrip("#")
    return tuple(  # type: ignore[return-value]
        int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)
    )


def _close(c1, c2, tol: float) -> bool:
    return c1 is not None and all(abs(a - b) <= tol for a, b in zip(c1, c2))


def _is_dark(color) -> bool:
    return color is not None and all(channel <= _DARK_CHANNEL_MAX for channel in color)


def _line_text(line: dict) -> str:
    return "".join(span["text"] for span in line["spans"]).strip()


def _lines_in(page: pymupdf.Page, x0: float, x1: float, y0: float, y1: float) -> list[str]:
    """中心落在矩形内的文本行（y 开区间语义由调用方传值控制）。"""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            bbox = pymupdf.Rect(line["bbox"])
            cx, cy = (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                text = _line_text(line)
                if text:
                    out.append(text)
    return out


def _find_body_bottom(drawings: list[dict], bar: pymupdf.Rect) -> float | None:
    """标题条下方第一条 x 对齐的深色细横线 = 框体部底边。"""
    tol_x = settings.box_x_tolerance_pt
    candidates = []
    for dr in drawings:
        if dr["type"] != "s" or not _is_dark(dr.get("color")):
            continue
        width = dr.get("width")
        if width is None or abs(width - 1.0) > 0.3:
            continue
        r = dr["rect"]
        if (
            r.height < 1
            and r.width >= (bar.x1 - bar.x0) - 10
            and abs(r.x0 - bar.x0) <= tol_x
            and abs(r.x1 - bar.x1) <= tol_x
            and bar.y1 + 1 < r.y0 <= bar.y1 + settings.box_bottom_search_pt
        ):
            candidates.append(r.y0)
    return min(candidates) if candidates else None


def detect_box_regions(page: pymupdf.Page) -> list[BoxRegion]:
    """检测单页的全部提示框区域，按 y 升序。"""
    if not settings.box_detect_enabled:
        return []
    page_no = page.number + 1
    target_fill = _hex_to_rgb01(settings.box_bar_fill_hex)
    tol = settings.box_color_tolerance
    tol_x = settings.box_x_tolerance_pt

    bars: list[pymupdf.Rect] = []
    for dr in page.get_drawings():
        if dr["type"] != "f" or not _close(dr.get("fill"), target_fill, tol):
            continue
        r = dr["rect"]
        if (
            settings.box_bar_min_width_pt < r.width < settings.box_bar_max_width_pt
            and settings.box_bar_min_height_pt < r.height < settings.box_bar_max_height_pt
        ):
            bars.append(r)

    regions: list[BoxRegion] = []
    for bar in sorted(bars, key=lambda rect: rect.y0):
        title_texts = _lines_in(page, bar.x0 - tol_x, bar.x1 + tol_x, bar.y0, bar.y1)
        title = "".join(title_texts).strip()
        if not TITLE_RE.match(title):
            continue  # 锚点校验：同色小节标题条在此被排除
        bottom = _find_body_bottom(page.get_drawings(), bar)
        if bottom is None:  # 跨页框首页：体部延伸到正文区底部（页脚之上）
            bottom = page.rect.y1 - 36
        content = _lines_in(page, bar.x0 - tol_x, bar.x1 + tol_x, bar.y1 + 1, bottom - 0.5)
        regions.append(
            BoxRegion(
                page_number=page_no,
                title=title,
                lines=[*title_texts, *content],
            )
        )
    return regions


def mark_box_lines(text: str, regions: list[BoxRegion]) -> tuple[str, int, int]:
    """把框内 markdown 行包成 ">" 引用块（归一化文本单调对齐）。

    标题行只按相等匹配（防止"请特别注意"这类行内提及误标）；
    内容行允许包含匹配（容纳 pymupdf4llm 偶发的粘行），并支持一行消费多个
    框内行。匹配不上的行保持原样——宁可漏标（补充块退化为部分内容），
    不误标正文。

    返回 (标记后的文本, 已标记行数, 框内行总数)。
    """
    expected: list[tuple[str, bool]] = []  # (归一化文本, 是否标题行)
    for region in regions:
        for index, line in enumerate(region.lines):
            norm = norm_text(line)
            if norm:
                expected.append((norm, index == 0))
    if not expected:
        return text, 0, 0

    lines = text.split("\n")
    pointer = 0
    marked = 0
    for i, line in enumerate(lines):
        if pointer >= len(expected):
            break
        line_norm = norm_text(line)
        if not line_norm:
            continue
        target, is_title = expected[pointer]
        if line_norm == target or (
            not is_title and len(target) >= 6 and target in line_norm
        ):
            lines[i] = "> " + line.lstrip()
            marked += 1
            pointer += 1
            while pointer < len(expected) and expected[pointer][0] in line_norm:
                pointer += 1  # 粘行：一行消费多个框内行
    return "\n".join(lines), marked, len(expected)
