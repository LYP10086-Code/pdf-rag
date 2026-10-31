# -*- coding: utf-8 -*-
"""表格结构重建：用单元格底纹矩形 + 文字坐标还原网格，输出干净 Markdown。

为什么不用 pymupdf4llm 自带的表格输出：
- 单元格内换行写成 <br>，一行 Markdown 被拆成多行，列数错位；
- 合并单元格（rowspan/colspan）没有表达，两级表头退化成空列；
- 跨页长表每页重复表头，同一张表被切成多张，行列语义断裂。

本模块只用确定性物理信号，与 style_headings 的标题探测互补：
- 表格区域：连续的单元格底纹填充路径（≥3 个矩形、≥150×15pt）；
- 行/列：单元格矩形的 y/x 边界聚类；表头 = 表格顶部独立填充带（≤3 行，
  且底色与表体不同）；
- 单元格文本：按坐标中心落入的单元格归属，多行按 CJK 规则拼接；
- 合并单元格：body 的 rowspan 向下填充，保证每一行自包含；表头 colspan
  提升为表头说明（不重复塞进每一列）；
- 跨页：表头签名 + 列边界 + 页尾/页首位置判定，合并成一张表。

页面文本中的表格先用 @@T{id}@@ 占位，全书处理完再统一替换为 Markdown，
这样跨页表可以把后续页的行追加到前页的表对象里。
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from app.ingestion.custom_parse.style_headings import norm_text

# ---- 表格区域探测参数（与 style_headings._table_bboxes 同口径）----
TABLE_MIN_RECTS = 3
TABLE_MIN_WIDTH = 150
TABLE_MIN_HEIGHT = 15
# 同一张表的多条填充路径：x 重叠比例、纵向间隙上限
PATH_X_OVERLAP = 0.5
PATH_Y_GAP = 140.0
# 表头与表体之间可能是未着色的白行（p42 间隙 92pt）；间隙里只有一两行
# 左对齐短文字时视为新表的小标题（p238"配置2"）
GAP_CAPTION_MAX_LINES = 1
# 表头 = 顶部独立填充带：纵向不超过整表高度的 45%，且最多 3 行
HEADER_MAX_HEIGHT_RATIO = 0.45
HEADER_MAX_ROW_BANDS = 3
# 边界聚类容差（pt）
EDGE_TOL = 1.5
CELL_EXPAND = 2.0
# 跨页续表：当前页表格必须贴近页首
NEAR_PAGE_TOP = 110.0
# 表格 Markdown 块与 PDF 表格区域的文本相似度下限
MATCH_MIN_RATIO = 0.5
# 粗体判定：PDF FontDescriptor 的 StemV 比 PyMuPDF 的 bold 位可靠。
# 本书常规宋体 StemV=68，表内粗宋（p233 的粗体 I）StemV=148。
BOLD_STEMV_THRESHOLD = 100
# 表头单位/口径标签（移到"表头说明"，避免挤占首列列名）
UNIT_LABEL_RE = re.compile(
    r"^[×xX]\s*1[,\s]*000|月数|[（(]\s*(?:公里|A|mm|cm|L|%|℃)\s*[)）]"
)
# 行尾是单位/右括号：下一行大概率是并列项（"…1L/桶；齿轮油…"），用分号
# 分隔。不要收"件/套/只"——EDR 正文里的"事件/条件"会被误伤成"事件；发生时"。
UNIT_END_RE = re.compile(r"(?:\dL|kg|\dg|mm|℃|桶|）|\))$")

TAG_RE = re.compile(r"<[^>]+>")
PLACEHOLDER_RE = re.compile(r"@@T(\d+)@@")
MD_SEP_RE = re.compile(r"\|(?:\s*:?-{3,}:?\s*\|)+")
ASCII_ALNUM_RE = re.compile(r"[0-9A-Za-z]")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class FillPath:
    rect: pymupdf.Rect
    cells: list[pymupdf.Rect]
    fill: tuple[float, ...] | None


@dataclass
class TextSpan:
    bbox: pymupdf.Rect
    text: str
    bold: bool = False


@dataclass
class TextLine:
    bbox: pymupdf.Rect
    text: str
    # 同一渲染行内的 span，用于判断一行是否跨了多个单元格；
    # bold 用于保留 p233 "I" 的粗体/常规两种含义
    spans: list[TextSpan] = field(default_factory=list)


@dataclass
class Table:
    page: int
    pages: list[int]
    header_note: str
    header: list[str]
    rows: list[list[str]]
    # 原始列边界（未删空图标列），用于跨页续表判定
    col_edges: list[float]
    signature: str
    y0: float
    y1: float
    page_height: float
    last_page: int
    last_y1: float
    last_page_height: float
    at_page_end: bool = False
    n_merged: int = 0

    def absorb(self, other: "Table") -> None:
        """把续页表的行并入本表（忽略续页重复表头）。"""
        self.rows.extend(other.rows)
        for p in other.pages:
            if p not in self.pages:
                self.pages.append(p)
        self.last_page = other.last_page
        self.last_y1 = other.last_y1
        self.last_page_height = other.last_page_height
        self.at_page_end = other.at_page_end
        self.n_merged += 1

    def to_markdown(self) -> str:
        if not self.header or not self.rows:
            return ""
        notes: list[str] = []
        if len(self.pages) > 1:
            notes.append(f"表格（跨第{min(self.pages)}至{max(self.pages)}页）：")
        if self.header_note:
            notes.append(f"表头说明：{self.header_note}")
        n = len(self.header)
        rows = [[(r[i] if i < len(r) else "") for i in range(n)] for r in self.rows]
        lines = list(notes)
        lines.append(_md_row([_md_cell(h) for h in self.header]))
        lines.append(_md_sep(n))
        lines.extend(_md_row([_md_cell(v) for v in r]) for r in rows)
        return "\n".join(lines)


# ---- 基础工具 ----

def _cluster(values: list[float], tol: float = EDGE_TOL) -> list[float]:
    """一维聚类，返回按序的簇中心（用于 x/y 边界去重）。"""
    if not values:
        return []
    vals = sorted(values)
    groups: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _bands(edges: list[float]) -> list[tuple[float, float]]:
    """边界序列 → 连续区间（行带）。"""
    return [
        (edges[i], edges[i + 1])
        for i in range(len(edges) - 1)
        if edges[i + 1] - edges[i] > 1.0
    ]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _is_ascii_alnum(ch: str) -> bool:
    return bool(ASCII_ALNUM_RE.match(ch))


def _smart_join(parts: list[str]) -> str:
    """单元格内多行文本拼接：CJK 行直接接上，ASCII 之后接词/中文补空格。"""
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out and UNIT_END_RE.search(out) and part[0] not in "。，；、,.;":
            out += "；"
        elif out and _is_ascii_alnum(out[-1]) and (
            _is_ascii_alnum(part[0]) or CJK_RE.match(part[0])
        ):
            out += " "
        out += part
    return out


def _md_cell(text: str) -> str:
    text = text.replace("|", "\\|")
    return re.sub(r"\s+", " ", text).strip()


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _md_sep(n: int) -> str:
    return "| " + " | ".join("---" for _ in range(n)) + " |"


def _color_diff(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> bool:
    if not a or not b or len(a) < 3 or len(b) < 3:
        return True
    return sum(abs(x - y) for x, y in zip(a[:3], b[:3])) > 0.08


def _norm_for_match(text: str) -> str:
    return norm_text(TAG_RE.sub("", text))


def _first_ref(value: str | None) -> int | None:
    if not value:
        return None
    m = re.search(r"(\d+)\s+0\s+R", value)
    return int(m.group(1)) if m else None


def _font_descriptor_ref(doc: pymupdf.Document, xref: int) -> int | None:
    """字体对象 → FontDescriptor 的 xref（兼容 Type0 的 DescendantFonts 数组）。"""
    kind, value = doc.xref_get_key(xref, "FontDescriptor")
    desc = _first_ref(value) if kind == "xref" else None
    if desc is not None:
        return desc
    kind, value = doc.xref_get_key(xref, "DescendantFonts")
    dref = _first_ref(value) if kind == "xref" else None
    if dref is None:
        return None
    obj = doc.xref_object(dref, compressed=False) or ""
    if obj.strip().startswith("["):
        dref = _first_ref(obj)
    if dref is None:
        return None
    kind2, value2 = doc.xref_get_key(dref, "FontDescriptor")
    return _first_ref(value2) if kind2 == "xref" else None


def _font_bold_map(page: pymupdf.Page) -> dict[str, bool]:
    """页面字体名 → 是否粗体。

    p233 的粗体 I/常规 I 使用两个不同字体（FZCSJW StemV=148 /
    FZSYJW StemV=68），PyMuPDF 的 flags bold 位对两者都为 False；
    必须读 PDF FontDescriptor 的 StemV 才能区分。
    """
    doc = page.parent
    if doc is None:
        return {}
    out: dict[str, bool] = {}
    for font in page.get_fonts(full=True):
        xref, name = font[0], font[3]
        base = name.split("+", 1)[-1]
        desc = _font_descriptor_ref(doc, xref)
        bold = False
        if desc is not None:
            obj = doc.xref_object(desc, compressed=False) or ""
            m = re.search(r"/StemV\s+(\d+)", obj)
            stemv = int(m.group(1)) if m else 0
            m2 = re.search(r"/FontWeight\s+(\d+)", obj)
            weight = int(m2.group(1)) if m2 else 0
            bold = stemv >= BOLD_STEMV_THRESHOLD or weight >= 600
        out[base] = bold
    return out


def _span_is_bold(span: dict, font_bold: dict[str, bool]) -> bool:
    if span.get("flags", 0) & 16:
        return True
    return font_bold.get(span.get("font", ""), False)


def _cell_markdown(items: list[TextLine]) -> str:
    """单元格多行文本；若所有非空 span 都是粗体，整体用 ** ** 标记。"""
    plain = _smart_join([ln.text for ln in items])
    spans = [
        s
        for ln in items
        for s in (ln.spans or [TextSpan(ln.bbox, ln.text, False)])
    ]
    nonempty = [s for s in spans if s.text.strip()]
    if nonempty and all(s.bold for s in nonempty):
        return f"**{plain}**"
    return plain


# ---- 表格区域探测 ----

def _fill_paths(page: pymupdf.Page) -> list[FillPath]:
    """页面内所有"像表格"的单元格底纹填充路径。"""
    paths: list[FillPath] = []
    for dr in page.get_drawings():
        if dr["type"] != "f":
            continue
        rects = [pymupdf.Rect(it[1]) for it in dr["items"] if it[0] == "re"]
        if len(rects) < TABLE_MIN_RECTS:
            continue
        bbox = pymupdf.Rect(rects[0])
        for r in rects[1:]:
            bbox.include_rect(r)
        if bbox.width < TABLE_MIN_WIDTH or bbox.height < TABLE_MIN_HEIGHT:
            continue
        paths.append(FillPath(bbox, rects, dr.get("fill")))
    return paths


def _gap_lines(page: pymupdf.Page, y0: float, y1: float) -> list[tuple[str, float, float]]:
    """纵向间隙内的文本行：[(text, size, x0)]（排除右侧页码/导航）。"""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for ln in block.get("lines", []):
            rect = pymupdf.Rect(ln["bbox"])
            text = "".join(s["text"] for s in ln["spans"]).strip()
            if not text or rect.x0 > page.rect.width - 40:
                continue
            cy = (rect.y0 + rect.y1) / 2
            if y0 - 2 <= cy <= y1 + 2:
                out.append((text, max(s["size"] for s in ln["spans"]), rect.x0))
    return out


def _gap_is_table_rows(
    page: pymupdf.Page,
    headings: list[tuple[float, str, str]],
    y0: float,
    y1: float,
    table_left: float,
) -> bool:
    """表头与表体之间的空隙是否仍是表格行（未着色的白行）。

    p42 表头在 y60-80、表体填充从 y173 才开始，中间 92pt 全是表格行文字，
    必须并成一张表；p238 的"配置2"紧贴表格左边缘、是间隙里的小标题，要切开。
    不用字号区分：p238 标题与 EDR 表格文字同为 10.5pt，字号不可靠。
    """
    if any(y0 - 2 <= y <= y1 + 2 for y, _t, _m in headings):
        return False
    lines = _gap_lines(page, y0, y1)
    if not lines:
        return True  # 空白间隙，仍视为同一张表的未着色行
    for text, _size, x0 in lines:
        # 紧贴表格左边缘的短行 = 小标题（"配置2"）；单元格文字通常有 5~17pt
        # 的左内边距（p290 的"外形尺寸"从 x51 开始，不会误判）
        if x0 <= table_left + 1.5 and len(text) <= 20:
            return False
    return True


def _group_paths(
    paths: list[FillPath], headings: list[tuple[float, str, str]], page: pymupdf.Page
) -> list[dict]:
    """按 x 重叠 + 纵向相邻把填充路径归并为表格区域。"""
    groups: list[dict] = []
    for path in sorted(paths, key=lambda p: (p.rect.y0, p.rect.x0)):
        candidates: list[tuple[float, dict]] = []
        for g in groups:
            bbox: pymupdf.Rect = g["bbox"]
            inter = _overlap(bbox.x0, bbox.x1, path.rect.x0, path.rect.x1)
            if inter / min(bbox.width, path.rect.width) < PATH_X_OVERLAP:
                continue
            gap = path.rect.y0 - bbox.y1
            if gap > PATH_Y_GAP:
                continue
            if gap > 2 and not _gap_is_table_rows(
                page, headings, bbox.y1, path.rect.y0, bbox.x0
            ):
                continue
            candidates.append((max(gap, 0.0), g))
        if candidates:
            # 优先并入纵向最近的组：p238 第二个表头被"配置2"切开后，后续
            # 表体路径必须跟第二个表头走，不能越过标题回到上一张表
            g = min(candidates, key=lambda x: x[0])[1]
            g["paths"].append(path)
            g["bbox"].include_rect(path.rect)
        else:
            groups.append({"paths": [path], "bbox": pymupdf.Rect(path.rect)})
    return groups


def _region_lines(
    page: pymupdf.Page, bbox: pymupdf.Rect, font_bold: dict[str, bool]
) -> list[TextLine]:
    """表格 bbox 内的渲染行，保留行内 span 列表。

    保留 span 是为了处理 pymupdf 的"同一基线合并多格"：p290 表头
    "产品型号名称BYD7008" 实为两个单元格的文字。归属时若整行 span 都落
    在同一个单元格，就按整行取（避免把一句话拆到两列）；只有一行跨了
    多个单元格时才按 span 拆开。
    """
    zone = bbox + (-4, -4, 4, 4)
    lines: list[TextLine] = []
    for block in page.get_text("dict")["blocks"]:
        for ln in block.get("lines", []):
            spans: list[TextSpan] = []
            for span in ln["spans"]:
                rect = pymupdf.Rect(span["bbox"])
                text = span["text"].strip()
                if not text or rect.y0 < 55:
                    continue
                if rect.x0 > page.rect.width - 40:
                    continue
                cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
                if zone.contains(pymupdf.Point(cx, cy)):
                    spans.append(
                        TextSpan(rect, text, _span_is_bold(span, font_bold))
                    )
            if not spans:
                continue
            line_rect = pymupdf.Rect(spans[0].bbox)
            for s in spans[1:]:
                line_rect.include_rect(s.bbox)
            line_text = "".join(s.text for s in spans)
            lines.append(TextLine(line_rect, line_text, spans))
    return lines


def find_table_regions(
    page: pymupdf.Page, headings: list[tuple[float, str, str]]
) -> list[dict]:
    """返回页面内的表格区域：{bbox, paths, lines}，按阅读序。"""
    groups = _group_paths(_fill_paths(page), headings, page)
    font_bold = _font_bold_map(page)
    regions = []
    for g in groups:
        lines = _region_lines(page, g["bbox"], font_bold)
        if not lines:
            continue
        regions.append({"bbox": g["bbox"], "paths": g["paths"], "lines": lines})
    regions.sort(key=lambda r: r["bbox"].y0)
    return regions


# ---- 网格重建 ----

def _split_header_body(
    region: dict,
) -> tuple[FillPath | None, list[FillPath]]:
    """区分表头填充带与表体填充路径。

    表头判据：位于表格顶部、纵向不超过整表 45%、最多 3 个行带，且底色与
    表体不同。p50/p51 那种"隔行底纹"首条路径纵贯整表，会被高度条件排除。
    """
    paths: list[FillPath] = region["paths"]
    top, height = region["bbox"].y0, region["bbox"].height
    if height <= 0:
        return None, paths
    for path in paths:
        if path.rect.y0 > top + 3 or path.rect.y1 > top + HEADER_MAX_HEIGHT_RATIO * height:
            continue
        edges = _cluster([c.y0 for c in path.cells] + [c.y1 for c in path.cells])
        if len(_bands(edges)) > HEADER_MAX_ROW_BANDS:
            continue
        others = [p for p in paths if p is not path]
        if others and all(_color_diff(path.fill, o.fill) for o in others):
            return path, others
    return None, paths


def _row_bands(
    header: FillPath | None,
    body_paths: list[FillPath],
    lines: list[TextLine],
    bbox: pymupdf.Rect,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """返回 (表头行带, 表体行带)。表体行带=边界连续区间（含未着色白行）。"""
    header_cells = header.cells if header else []
    body_cells = [c for p in body_paths for c in p.cells]

    header_edges = _cluster([c.y0 for c in header_cells] + [c.y1 for c in header_cells])
    header_bands = _bands(header_edges) if header_edges else []

    body_edges = _cluster([c.y0 for c in body_cells] + [c.y1 for c in body_cells])
    body_bands = _bands(body_edges)
    if header and body_cells:
        hb = max(c.y1 for c in header_cells)
        bt = min(c.y0 for c in body_cells)
        if bt - hb > 2 and any(
            hb <= (line.bbox.y0 + line.bbox.y1) / 2 <= bt for line in lines
        ):
            body_bands.insert(0, (hb, bt))
    if not body_bands:
        body_bands = [(bbox.y0, bbox.y1)]
    return header_bands, body_bands


def _modal_x_edges(
    body_cells: list[pymupdf.Rect], body_bands: list[tuple[float, float]]
) -> list[float]:
    """从"完整落在一行内"的单元格中取最常见的列边界模式。

    合并单元格（rowspan）不完整落在任何一行，会被自然排除；这避免了
    p290 中"质量参数"竖排标签产生的伪列边界。
    """
    patterns: Counter[tuple[float, ...]] = Counter()
    for y0, y1 in body_bands:
        full = [
            c
            for c in body_cells
            if abs(c.y0 - y0) <= 2 and abs(c.y1 - y1) <= 2 and c.width > 2
        ]
        if len(full) < 2:
            continue
        edges = _cluster([c.x0 for c in full] + [c.x1 for c in full])
        if len(edges) >= 2:
            patterns[tuple(round(e, 1) for e in edges)] += 1
    if not patterns:
        return []
    return list(max(patterns.items(), key=lambda kv: (len(kv[0]), kv[1]))[0])


def _column_edges(
    header_cells: list[pymupdf.Rect],
    body_cells: list[pymupdf.Rect],
    body_bands: list[tuple[float, float]],
) -> list[float] | None:
    """列边界 = 表体众数模式的边界 ∪ 被表体实际使用的表头边界。

    p233 表头多出一个"安培/里程"单位列，88.3 这条边界只存在于表头，会被
    过滤掉；p105 的表头首列边界虽然只在合并单元格上出现，但被表体使用，
    因此保留。
    """
    used = _cluster([c.x0 for c in body_cells] + [c.x1 for c in body_cells])
    modal = _modal_x_edges(body_cells, body_bands)
    header_edges = _cluster([c.x0 for c in header_cells] + [c.x1 for c in header_cells])
    keep_header: list[float] = []
    last = len(header_edges) - 1
    for i, h in enumerate(header_edges):
        used_edge = any(abs(h - u) <= 2 for u in used)
        outer = i == 0 or i == last
        # 只被表头使用、且两侧列宽都很窄的边界，多半是 p233 那种单位子列；
        # p291 的型号列虽然也只出现在表头（表体是跨列合并单元格），但列宽
        # 正常，必须保留，否则 6 个型号列会被压成 2 列。
        narrow = False
        if not outer:
            narrow = min(h - header_edges[i - 1], header_edges[i + 1] - h) < 35
        if used_edge or outer or not narrow:
            keep_header.append(h)
    filtered = _cluster(keep_header)
    if filtered:
        # 表头列边界优先：p290 的竖排"质量参数"会让表体产生 74.3/98.4
        # 两条伪边界，按表头 7 列合并后才是正确网格。只有表头单元格跨
        # 多个表体列（表头列数少于表体众数）时才补入表体边界。
        header_cols = len(filtered) - 1
        modal_cols = len(modal) - 1 if modal else 0
        edges = _cluster(filtered + modal) if modal_cols > header_cols else filtered
    else:
        edges = _cluster(modal or used)
    if len(edges) < 2:
        return None
    return edges


def _anchor(
    cell: pymupdf.Rect, bands: list[tuple[float, float]], edges: list[float]
) -> tuple[int, int, int, int] | None:
    """单元格矩形 → (起始行, 起始列, 结束行, 结束列)（闭区间）。"""
    rows = [
        i
        for i, (a, b) in enumerate(bands)
        if _overlap(a, b, cell.y0, cell.y1) > 1
    ]
    cols = [
        i
        for i in range(len(edges) - 1)
        if _overlap(edges[i], edges[i + 1], cell.x0, cell.x1) > 1
    ]
    if not rows or not cols:
        return None
    return rows[0], cols[0], rows[-1], cols[-1]


def _find_cell_rect(bbox: pymupdf.Rect, cells: list[pymupdf.Rect]) -> pymupdf.Rect | None:
    """矩形中心落在哪个单元格（取面积最小的，处理嵌套/外扩重叠）。"""
    point = pymupdf.Point((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)
    best: pymupdf.Rect | None = None
    best_area = float("inf")
    for cell in cells:
        zone = cell + (-CELL_EXPAND, -CELL_EXPAND, CELL_EXPAND, CELL_EXPAND)
        if zone.contains(point):
            area = cell.width * cell.height
            if area < best_area:
                best, best_area = cell, area
    return best


def _band_index(cy: float, bands: list[tuple[float, float]]) -> int | None:
    for i, (a, b) in enumerate(bands):
        if a <= cy <= b:
            return i
    return None


def _col_index(bbox: pymupdf.Rect, edges: list[float]) -> int | None:
    best: int | None = None
    best_ov = 0.0
    for i in range(len(edges) - 1):
        ov = _overlap(edges[i], edges[i + 1], bbox.x0, bbox.x1)
        if ov > best_ov:
            best, best_ov = i, ov
    return best


def _parse_region(region: dict, page_no: int, page_height: float) -> Table | None:
    header, body_paths = _split_header_body(region)
    header_cells = header.cells if header else []
    body_cells = [c for p in body_paths for c in p.cells]
    lines: list[TextLine] = region["lines"]
    bbox: pymupdf.Rect = region["bbox"]

    header_bands, body_bands = _row_bands(header, body_paths, lines, bbox)
    edges = _column_edges(header_cells, body_cells, body_bands)
    if edges is None:
        return None

    # 文本行 → 单元格。整行 span 都落在同一格 → 按整行取（避免把一句话
    # 拆到两列）；一行跨多个格 → 按 span 拆开（p290 表头）；没有任何格
    # 命中（未着色白行）→ 按整行 bbox 归到行列带，保持句子完整。
    all_cells = header_cells + body_cells
    cell_text: dict[int, list[TextLine]] = {}
    extra: dict[tuple[int, int], list[TextLine]] = {}
    for line in lines:
        spans = line.spans or [TextSpan(line.bbox, line.text)]
        span_cells = [_find_cell_rect(s.bbox, all_cells) for s in spans]
        hit = {id(c) for c in span_cells if c is not None}
        if len(hit) == 1:
            cell = next(c for c in span_cells if c is not None)
            cell_text.setdefault(id(cell), []).append(line)
            continue
        if hit:
            for span, cell in zip(spans, span_cells):
                one = TextLine(span.bbox, span.text, [span])
                if cell is not None:
                    cell_text.setdefault(id(cell), []).append(one)
                    continue
                cy = (span.bbox.y0 + span.bbox.y1) / 2
                r = _band_index(cy, body_bands)
                c = _col_index(span.bbox, edges)
                if r is not None and c is not None:
                    extra.setdefault((r, c), []).append(one)
            continue
        cy = (line.bbox.y0 + line.bbox.y1) / 2
        r = _band_index(cy, body_bands)
        c = _col_index(line.bbox, edges)
        if r is not None and c is not None:
            extra.setdefault((r, c), []).append(line)

    def cell_str(
        cell: pymupdf.Rect, use_bottom: bool = False, mark_bold: bool = True
    ) -> str:
        items = cell_text.get(id(cell), [])
        if use_bottom and items:
            # 跨多行表头单元格（p233 的"保养时间间隔/保养项目"）取底部
            # 40% 的文字，让列名落在"保养项目"而不是两个竖排标签全拼
            bottom_zone = cell.y1 - 0.4 * cell.height
            bottom = [ln for ln in items if (ln.bbox.y0 + ln.bbox.y1) / 2 >= bottom_zone]
            if bottom:
                items = bottom
        return _cell_markdown(items) if mark_bold else _smart_join(
            [ln.text for ln in items]
        )

    # 表头：colspan>1 的组标题进"表头说明"；单位标签也进说明；
    # 叶列标题进列名
    header_parts: list[list[str]] = [[] for _ in range(len(edges) - 1)]
    shared: list[str] = []
    unit_notes: list[str] = []
    seen_shared: set[str] = set()
    for cell in header_cells:
        anchor = _anchor(cell, header_bands, edges)
        if anchor is None:
            continue
        r0, c0, r1, c1 = anchor
        # 表头本来就是加粗字体，不把 ** ** 塞进列名
        text = cell_str(cell, use_bottom=(r1 > r0), mark_bold=False)
        if not text:
            continue
        if c1 > c0:
            key = norm_text(text)
            if key and key not in seen_shared:
                seen_shared.add(key)
                shared.append(text)
        elif c0 == 0 and UNIT_LABEL_RE.search(text):
            if text not in unit_notes:
                unit_notes.append(text)
        elif text not in header_parts[c0]:
            header_parts[c0].append(text)

    # 表体：rowspan 向下填充，保证每一行自包含；colspan 只写最左列
    n_rows, n_cols = len(body_bands), len(edges) - 1
    parts: list[list[list[str]]] = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    extra_center: dict[tuple[int, int], float] = {}
    for cell in body_cells:
        text = cell_str(cell)
        if not text:
            continue
        anchor = _anchor(cell, body_bands, edges)
        if anchor is None:
            continue
        r0, c0, r1, c1 = anchor
        for r in range(r0, min(r1, n_rows - 1) + 1):
            if c0 >= 1 and c1 > c0:
                # 数据区内的跨列合并单元格（p290"轮距 1580/1580"）：
                # 值适用于所有配置列，逐列复制，避免看起来只属于某一列
                for c in range(c0, c1 + 1):
                    parts[r][c].append(text)
            else:
                parts[r][c0].append(text)
    for (r, c), items in extra.items():
        if 0 <= r < n_rows and 0 <= c < n_cols:
            parts[r][c].append(_cell_markdown(items))
            extra_center[(r, c)] = sum(
                (ln.bbox.x0 + ln.bbox.x1) / 2 for ln in items
            ) / len(items)

    # 规格表里"适用于所有配置"的单元格在 PDF 中跨多列居中，但该行没有
    # 底纹可取合并跨度（p290 外形尺寸/轮距/轴距）。若整行只有一个数据值
    # 且它正好居中于数据区，则复制到每个数据列，避免值看起来只属于某一列。
    if n_cols >= 4:
        data_mid = (edges[1] + edges[-1]) / 2
        for r in range(n_rows):
            data_cols = [c for c in range(1, n_cols) if parts[r][c]]
            if len(data_cols) != 1:
                continue
            c = data_cols[0]
            center = extra_center.get((r, c))
            if center is None or abs(center - data_mid) > 6:
                continue
            value = list(parts[r][c])
            for dc in range(1, n_cols):
                parts[r][dc] = list(value)

    # 多级表头用 "/" 连接（p233 "5/3" = 5×1,000 公里 / 3 个月），
    # 同一单元格内的多行文字（型号名 BYD7008 BEVB）保持空格
    header = ["/".join(dict.fromkeys(p)) for p in header_parts]
    header_note = "；".join(shared)
    if unit_notes:
        header_note = "；".join(
            filter(None, [header_note, "单位/口径：" + "、".join(unit_notes)])
        )
    if not any(header):
        header = [shared[0]] * n_cols if shared else [""] * n_cols
        if shared and unit_notes:
            header_note = "单位/口径：" + "、".join(unit_notes)

    rows: list[list[str]] = []
    for r in range(n_rows):
        values = [" ".join(dict.fromkeys(p)) for p in parts[r]]
        if any(v.strip() for v in values):
            rows.append(values)
    if not rows:
        return None

    # 首列只有一个取值、且部分行空着（p106 无底纹的"充电中途停止充电"
    # 竖跨 4 行）：整页内把这个值补到所有行。多标签表不启用，避免把不同
    # 分组标签错误平铺。
    if n_cols >= 3:
        firsts = {row[0] for row in rows if row[0]}
        if len(firsts) == 1 and any(not row[0] for row in rows):
            only = next(iter(firsts))
            for row in rows:
                if not row[0]:
                    row[0] = only
        elif len(firsts) > 1:
            # p36 的"0组/0+组"竖跨两行，但无底纹行不会画出合并单元格：
            # 首列空格按上一个非空值向下填充
            last = ""
            for row in rows:
                if row[0]:
                    last = row[0]
                elif last:
                    row[0] = last

    # 删掉"无文字且窄"的图标列（p50/p51 指示灯表：图标列不参与文本检索）
    keep = []
    for c in range(n_cols):
        width = edges[c + 1] - edges[c]
        if any(row[c] for row in rows) or header[c] or width >= 80:
            keep.append(c)
    if not keep:
        return None
    header = [header[c] for c in keep]
    rows = [[row[c] for c in keep] for row in rows]

    signature = norm_text(" ".join(header)) if any(header) else ""
    if not signature and header_note:
        signature = norm_text(header_note)

    return Table(
        page=page_no,
        pages=[page_no],
        header_note=header_note,
        header=header,
        rows=rows,
        col_edges=edges,
        signature=signature,
        y0=bbox.y0,
        y1=bbox.y1,
        page_height=page_height,
        last_page=page_no,
        last_y1=bbox.y1,
        last_page_height=page_height,
    )


# ---- 页面替换与跨页合并 ----

def _find_md_table_blocks(text: str) -> list[tuple[int, int]]:
    lines = text.split("\n")
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        j = i
        while j < len(lines):
            if lines[j].lstrip().startswith("|"):
                j += 1
                continue
            if not lines[j].strip() and j + 1 < len(lines) and lines[j + 1].lstrip().startswith("|"):
                j += 1
                continue
            break
        end = j
        while end > i and not lines[end - 1].strip():
            end -= 1
        blocks.append((i, end))
        i = end
    return blocks


def _can_absorb(cur: Table, prev: Table, page_no: int) -> bool:
    if page_no != prev.last_page + 1:
        return False
    # 上一页的表格必须是该页最后的正文内容（p40-45 的续表页往往只有表，
    # 表本身不必贴到页底）；p35 的表后面还有"注："，因此不会误并)
    if not prev.at_page_end:
        return False
    if cur.y0 > NEAR_PAGE_TOP:
        return False
    if cur.signature != prev.signature:
        return False
    if len(cur.col_edges) != len(prev.col_edges):
        return False
    return all(abs(a - b) <= 3 for a, b in zip(cur.col_edges, prev.col_edges))


def replace_tables_in_page(
    page: pymupdf.Page,
    page_no: int,
    md_text: str,
    registry: list[Table],
    last_table: Table | None,
    headings: list[tuple[float, str, str]],
) -> tuple[str, Table | None]:
    """把页面 Markdown 中的表格块替换为 @@T{id}@@ 占位（或并入上一页续表）。

    只替换"能匹配上 PDF 表格区域"的 Markdown 块：pymupdf4llm 有时把表格
    当图片文本输出（如 p194 第一张表），这类内容没有 | 块，保持原样，避免
    与重建表重复。
    """
    blocks = _find_md_table_blocks(md_text)
    if not blocks:
        return md_text, last_table
    regions = find_table_regions(page, headings)

    lines = md_text.split("\n")
    scores: list[tuple[float, int, int]] = []
    block_texts = ["\n".join(lines[a:b]) for a, b in blocks]
    region_texts = [" ".join(l.text for l in r["lines"]) for r in regions]
    for bi, btext in enumerate(block_texts):
        for ri, rtext in enumerate(region_texts):
            ratio = difflib.SequenceMatcher(
                None, _norm_for_match(btext), _norm_for_match(rtext)
            ).ratio()
            scores.append((ratio, bi, ri))
    scores.sort(reverse=True)

    matched_block: dict[int, int] = {}
    matched_region: dict[int, int] = {}
    for ratio, bi, ri in scores:
        if ratio < MATCH_MIN_RATIO:
            break
        if bi in matched_block or ri in matched_region:
            continue
        matched_block[bi] = ri
        matched_region[ri] = bi

    replacements: dict[int, list[str]] = {}
    for bi in sorted(matched_block):
        ri = matched_block[bi]
        table = _parse_region(regions[ri], page_no, page.rect.height)
        if table is None:
            continue
        # 该表之后页面是否还有正文：续表只允许从"上一页最后一块"接续
        table.at_page_end = not any(
            line.strip() for line in lines[blocks[bi][1]:]
        )
        if last_table is not None and _can_absorb(table, last_table, page_no):
            last_table.absorb(table)
            replacements[bi] = []
        else:
            registry.append(table)
            last_table = table
            replacements[bi] = ["", f"@@T{len(registry) - 1}@@", ""]

    # 未匹配 PDF 表格区域的单列表格块（p141 的"温馨提示"提示框被
    # pymupdf4llm 当成了 |温馨提示| 表格）：还原为普通文本行，交给提示框
    # 抽取；真正的表格至少两列，不会走这条。
    for bi, (start, end) in enumerate(blocks):
        if bi in matched_block:
            continue
        first = next(
            (ln for ln in lines[start:end] if ln.lstrip().startswith("|")), ""
        )
        if not first or first.count("|") > 2:
            continue
        flat: list[str] = []
        for ln in lines[start:end]:
            s = ln.strip()
            if not s or MD_SEP_RE.fullmatch(s):
                continue
            if s.startswith("|"):
                cells = [c.strip() for c in s.strip("|").split("|")]
                s = " ".join(c for c in cells if c)
            if s:
                flat.append(s)
        replacements[bi] = flat

    for bi in sorted(replacements, reverse=True):
        start, end = blocks[bi]
        lines[start:end] = replacements[bi]
    return "\n".join(lines), last_table


def resolve_placeholders(text: str, registry: list[Table]) -> str:
    return PLACEHOLDER_RE.sub(
        lambda m: registry[int(m.group(1))].to_markdown()
        if int(m.group(1)) < len(registry)
        else "",
        text,
    )
