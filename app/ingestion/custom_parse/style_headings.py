# -*- coding: utf-8 -*-
"""样式签名标题探测：用 PDF 物理信号（字号/颜色/矢量路径）确定性识别标题层级。

背景：本书的标题没有可靠的字体元数据（加粗为"文字转曲"伪加粗，字体子集
二进制相同，文本层与正文零差异），pymupdf4llm 的字号频率法无法正确分层。
但每种层级都有唯一的物理签名（全书扫描验证，零噪声）：

    层级  视觉特征                     物理签名                        Markdown
    L1    章扉页大标题（安全）          STXihei 18pt 黑                 #
    L2    节标题（1-1 座椅安全带）       页眉带 y<35（extract_section_titles 负责，不在此注入）
    L3    深绿条白加粗（安全带简介）     STXihei 14pt + 白色             ###
    L4    浅绿条黑加粗（充电方式）       STXihei 12pt（正文区）          ####
    L5    正文转曲加粗（温度调节）       与正文同款文字 + 字形轮廓路径     #####

提示框标题（警告/注意/温馨提示，STXihei 14pt + 黑色）**不作为标题**：
语义上它们只是正文中的着重提醒，与其所属步骤的正文一体，独立成块会把
框后无标题正文错误吸入框的 fragment（跨页时尤其明显）。因此不进入标题
清单，其 `#` 标记由注入时的统一剥离降为普通文本行，标题文字保留在正文
中（"警告"等词的检索语义不丢失）。

L5 探测原理：正文加粗由"文字转曲"实现——每个加粗字符的轮廓以小型 stroke
路径叠画在正常文本之下（正文区域路径数为 0，加粗行每字一条），检测行
bbox 内的轮廓路径数即可判定。非加粗的 ■ 格式标签（如"主驾温度调节按键"）
与正文同字重，不会被误判（这是设计行为：它们保持列表项身份）。

输出：build_heading_index(doc) -> {page(1-based): [(y, text, marker), ...]}（按 y 升序）
"""
import re

import pymupdf

# 书籍前部（封面/版权/前言），无正文标题
FRONT_MATTER_PAGES = 4
# 页眉带上界（页眉内容 y<35，正文从 y>60 开始）
HEADER_BAND_Y = 35
# 字形轮廓路径的尺寸范围（单个汉字转曲轮廓约 8~12pt 宽高）
OUTLINE_W = (3.0, 30.0)
OUTLINE_H = (3.0, 16.0)
# 加粗判定：轮廓路径数下限，以及路径数对 CJK 字数的最低覆盖率
BOLD_MIN_HITS = 2
BOLD_MIN_RATIO = 0.6
# L5 转曲加粗只认正文字号（表格文字被缩放为 8.5~9pt，图注 10.2~10.4pt，
# 以此排除表格/图内文字，替代 find_tables——后者在本书部分页会触发 MuPDF 原生崩溃）
BOLD_SIZE_RANGE = (9.8, 11.2)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# markdown 伪影（先于通用归一化剥离，否则 <u> 的字母 u 会残留干扰匹配）
MARKUP_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|</?(?:mark|u|s)>|~~|`+")
# 通用归一化：仅保留字母数字与 CJK
NORM_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")


def norm_text(s: str) -> str:
    """文本归一化（标题对齐匹配用）：剥离 markdown 伪影与非字母数字字符。"""
    return NORM_RE.sub("", MARKUP_RE.sub("", s))


def _glyph_outline_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """提取字形轮廓大小的 stroke 路径矩形（文字转曲的物理痕迹）。"""
    rects = []
    for dr in page.get_drawings():
        if dr["type"] not in ("s", "fs"):
            continue
        r = dr["rect"]
        if OUTLINE_W[0] < r.width < OUTLINE_W[1] and OUTLINE_H[0] < r.height < OUTLINE_H[1]:
            rects.append(r)
    return rects


def _table_bboxes(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """基于单元格底纹填充检测表格区域（不使用 find_tables——其在本书部分页
    会触发 MuPDF 原生崩溃）。

    本书的表格无边框格线，用成组的单元格底色矩形绘制：一个填充路径含
    ≥3 个 're' 矩形项、整体宽≥150pt 高≥15pt 即一张表（如 p36 质量组表
    表头 7 格橙色 + 表体 33 格浅黄；p39 EDR 表为单行表头，高仅 21pt）。
    返回全部单元格矩形（略外扩），命中单元格的文本行视为表内内容，
    不参与标题探测。

    不会误伤：绿条标题/提示框底色均为单个矩形（<3 项），页面底色仅 1 项。
    """
    cells: list[pymupdf.Rect] = []
    for dr in page.get_drawings():
        r = dr["rect"]
        n_re = sum(1 for it in dr["items"] if it[0] == "re")
        if dr["type"] == "f" and n_re >= 3 and r.width >= 150 and r.height >= 15:
            for it in dr["items"]:
                if it[0] == "re":
                    cells.append(pymupdf.Rect(it[1]) + (-2, -2, 2, 2))
    return cells


def _line_text(line: dict) -> str:
    return "".join(s["text"] for s in line["spans"]).strip()


def _dominant_span(line: dict) -> dict:
    """行内字符最多的 span（决定该行的字体/字号/颜色）。"""
    return max(line["spans"], key=lambda s: len(s["text"]))


def _bold_hit_count(bbox: pymupdf.Rect, outlines: list[pymupdf.Rect]) -> int:
    """统计中心落入行 bbox（外扩 2pt）的轮廓路径数。"""
    zone = bbox + (-2, -2, 2, 2)
    n = 0
    for r in outlines:
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if zone.contains(pymupdf.Point(cx, cy)):
            n += 1
    return n


def detect_page_headings(page: pymupdf.Page) -> list[tuple[float, str, str]]:
    """探测单页标题，返回 [(y, text, markdown_marker), ...] 按 y 升序。"""
    page_no = page.number + 1
    if page_no <= FRONT_MATTER_PAGES:
        return []
    outlines = _glyph_outline_rects(page)
    tables = _table_bboxes(page)
    heads: list[tuple[float, str, str]] = []

    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            bbox = pymupdf.Rect(line["bbox"])
            text = _line_text(line)
            if not text or bbox.y0 < HEADER_BAND_Y:
                continue  # 页眉带（L2 由 extract_section_titles 单独负责）
            if bbox.x0 > page.rect.width - 40:
                continue  # 右侧竖排章节导航标签（如"5 车内装置"）
            if re.search(r"\.{3}|\u2026{2}", text):
                continue  # 目录点线行（页 5-8 总目录 + 章扉页迷你目录）
            if any(t.intersects(bbox) for t in tables):
                continue  # 表格内容（表头/单元格文本，一律按正文存储）
            span = _dominant_span(line)
            size, color = span["size"], span["color"]
            n_cjk = len(CJK_RE.findall(text))

            if size > 16:  # 18pt 章扉页标题
                marker = "#"
            elif 13 <= size <= 15 and color == 0xFFFFFF:  # 白字 = 深绿条标题
                marker = "###"
            elif 13 <= size <= 15 and color != 0xFFFFFF:
                continue  # 黑字 14pt = 提示框标题（警告/注意/温馨提示）→ 按正文处理，不作为标题
            elif 11.5 <= size < 13:  # 12pt = 浅绿条标题
                marker = "####"
            else:
                # 正文字号（~10.5pt）：仅当存在转曲轮廓路径时才是加粗标题；
                # 字号范围过滤掉缩放的表格文字（8.5~9pt）与图注（10.2~10.4pt 边界外）
                if not (BOLD_SIZE_RANGE[0] <= size <= BOLD_SIZE_RANGE[1]):
                    continue
                hits = _bold_hit_count(bbox, outlines) if outlines else 0
                if hits < BOLD_MIN_HITS or hits < BOLD_MIN_RATIO * max(n_cjk, 1):
                    continue
                # 加粗强调句（句号结尾且较长，如"如果一切正常，则大可放心享受
                # 驾驶的乐趣。"）不是标题，跳过；标题式加粗以冒号/无标点收尾
                if len(text) >= 10 and text.endswith(("。", "！", "？", "；")):
                    continue
                marker = "#####"
            heads.append((bbox.y0, text, marker))

    heads.sort(key=lambda h: h[0])
    return heads


def build_heading_index(doc: pymupdf.Document) -> dict[int, list[tuple[float, str, str]]]:
    """全书标题索引：{page(1-based): [(y, text, marker), ...]}。"""
    index: dict[int, list[tuple[float, str, str]]] = {}
    for page in doc:
        heads = detect_page_headings(page)
        if heads:
            index[page.number + 1] = heads
    return index
