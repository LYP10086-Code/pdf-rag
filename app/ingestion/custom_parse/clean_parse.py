# -*- coding: utf-8 -*-
"""PDF 清洗：样式签名标题注入 + 每页章节归属 + 伪影清理。

思路（标题全部来自确定性物理信号，详见 style_headings.py）：
1. 页眉带（顶部 55pt）单独提取章节标题（前向填充），作为分块的分节根。
2. 正文用 pymupdf4llm 的 header=False / footer=False 解析（保留其列表/
   段落能力），但**不采信它猜的标题**——先剥掉它输出的全部 # 前缀。
3. style_headings 按物理签名探测各级标题（18pt 章标题 / 14pt 白字 / 12pt 绿条 /
   转曲加粗 / 14pt 提示框），按 (page, y) 坐标对齐注入正确的 # 层级。
4. 表格块不采信 pymupdf4llm 的 <br> 输出：table_parse 用单元格底纹 + 文本坐标
   重建网格、填充合并单元格、合并跨页续表（见 table_parse.py / design.md 4.5）。
5. 清理 Markdown 伪影（~~删除线~~、<mark>、<u>、Q/█ 图标残留）。
6. 输出 clean_pages.jsonl（每页: 页码 + 章节标题 + 正文），供
   MarkdownHeaderTextSplitter 分块时把标题链写入 chunk metadata。

"""
import re

import pymupdf

from app.core.config import settings
from app.ingestion.custom_parse import box_regions
from app.ingestion.custom_parse.style_headings import build_heading_index, norm_text
from app.ingestion.custom_parse.table_parse import (
    replace_tables_in_page,
    resolve_placeholders,
)

HEADER_BAND = 55  # 页眉带高度（pt），实测页眉内容均在 y<40，正文从 y>60 开始


def extract_section_titles(doc: pymupdf.Document) -> list[str]:
    """逐页解析页眉带，返回每页所属章节标题（前向填充）。"""
    cjk = r"[\u4e00-\u9fff/]"
    space_join = re.compile(rf"(?<={cjk})[ \t]+(?={cjk})")  # "充/ 放电" -> "充/放电"
    titles: list[str] = []
    last = ""
    for page in doc:
        raw = page.get_text("text", clip=pymupdf.Rect(0, 0, page.rect.width, HEADER_BAND))
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        title_parts = [line for line in lines if not re.fullmatch(r"\d+", line)]
        if title_parts:  # 页眉带出现新章节标题时更新
            last = space_join.sub("", " ".join(title_parts))
        titles.append(last)
    return titles


def clean_markdown_artifacts(text: str) -> str:
    """清理 pymupdf4llm 输出中的样式伪影（不含图标字符）。"""
    text = text.replace("~~", "")  # 加粗中文被误判为删除线
    text = re.sub(r"</?(?:mark|u)>", "", text)  # 高亮/下划线标签
    text = re.sub(r"`+([“”])`+", r"\1", text)  # 反引号包裹的引号
    text = re.sub(r"`+([“”])", r"\1", text)
    text = re.sub(r"([“”])`+", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # 压缩连续空行
    # 合并 CJK 字符间的空格（PDF 字体间距导致，如 "座 椅 侧" -> "座椅侧"）
    cjk = r"[\u4e00-\u9fff/]"
    text = re.sub(rf"(?<={cjk})[ \t]+(?={cjk})", "", text)
    return text.strip()


# ---- 标题注入：用 style_headings 的物理签名探测结果替换 pymupdf4llm 的猜测 ----

# 前缀匹配允许的归一化残差上限（行内图标/图片标记剥离后的残留字符）
MATCH_RESIDUE = 12
# 归一化保留字符（与 style_headings.norm_text 一致）
_NORM_KEEP = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")


def _norm_map(raw: str) -> tuple[str, list[int]]:
    """返回 (归一化串, 每个 norm 字符在 raw 中的下标)，用于原位切分。"""
    chars, idxs = [], []
    for i, ch in enumerate(raw):
        if _NORM_KEEP.match(ch):
            chars.append(ch)
            idxs.append(i)
    return "".join(chars), idxs


def _split_glued(lines: list[str], hnorms: list[tuple[str, str, str]]) -> list[str]:
    """预切分粘连行：一行内按 y 序出现 ≥2 个标题时按原位切开。

    pymupdf4llm 偶发把相邻标题缝进同一行（如 "- 通风按键 █ 按下 " █ 出风模式"），
    按各标题的归一化位置切分并就地标注，切分碎片（"按下 "" 之类残渣）随
    后续匹配被标题行替换或保留为碎行。

    粘连标题的尾部特征（2026-10-30 p82 碎片化修复）：标题词后随内容
    （跳过空白）必须是图标残渣（█/Q）或行尾。后随 CJK/引号/括号/数字的
    是正文中的行内提及——如"自动控制小灯和近光灯的开启或关闭"
    （p82，行内加粗词与本页 L5 标题同名）、"" 解锁 " 按键"（p65，按键名
    加引号）、"预测性碰撞报警 (PCW)"（p166，名称+缩写）——不是缝进本行
    的标题，禁止切分。行尾单个真标题不经此处，由注入主循环的"行尾后缀"
    规则拆分（前缀正文无损保留）。
    """
    out: list[str] = []
    all_hnorms = {hnorm for hnorm, _t, _m in hnorms}
    for ln in lines:
        s, idxs = _norm_map(ln)
        if len(s) < 4 or not hnorms:
            out.append(ln)
            continue
        if s in all_hnorms:
            out.append(ln)  # 整行就是某个标题（如"进入功能和启动功能"），禁止切分
            continue
        found: list[tuple[int, int, str, str]] = []  # (raw_start, raw_end, text, marker)
        for hnorm, text, marker in hnorms:
            pos = s.find(hnorm)
            if pos < 0:
                continue
            end = idxs[pos + len(hnorm) - 1] + 1
            rest = ln[end:].lstrip(" \t")
            if rest and rest[0] not in "█Q":
                continue  # 行内提及（后随正文/引号/括号等），不是粘连标题
            found.append((idxs[pos], end, text, marker))
        found.sort()
        # 去重/去重叠，且要求 ≥2 个不同标题
        merged: list[tuple[int, int, str, str]] = []
        for f in found:
            if merged and f[0] < merged[-1][1]:
                continue  # 与前一个重叠，放弃后者
            merged.append(f)
        if len(merged) < 2:
            out.append(ln)
            continue
        cur = 0
        for rs, re_, text, marker in merged:
            seg = ln[cur:rs]
            if seg.strip():
                out.append(seg)
            out.append(f"{marker} {text}")
            cur = re_
        if ln[cur:].strip():
            out.append(ln[cur:])
    return out


def inject_headings(
    md_text: str, heads: list[tuple[float, str, str]]
) -> tuple[str, int, int]:
    """把样式探测的标题注入 pymupdf4llm 的页面 markdown。

    步骤：
    1. 剥掉 pymupdf4llm 自己标注的全部 # 前缀——其字号频率猜测不分层
       （14pt 白字/黑字、12pt 绿条混为一级，10.5pt 转曲加粗完全漏检），不采信。
    2. 粘连行预切分（一行含 ≥2 个标题）。
    3. 按 y 升序逐标题匹配（归一化，两序列同为阅读序、指针单调推进）：
       精确相等 > 行首前缀（残差≤12 替换整行；行首后接空格则原位拆分）
       > 行尾后缀（标题被缝进上一行行尾，拆分保留前缀正文）。

    返回 (注入后的文本, 匹配数, 探测总数)。
    """
    lines = [re.sub(r"^#{1,6}[ \t]+", "", ln) for ln in md_text.split("\n")]
    hnorms = [(norm_text(t), t, m) for _y, t, m in heads if len(norm_text(t)) >= 2]
    lines = _split_glued(lines, hnorms)

    mi = 0  # 已消费的 markdown 行指针（单调推进）
    matched = 0
    for hnorm, text, marker in hnorms:
        cand = None  # (action, j)  action: exact / prefix / split
        for j in range(mi, len(lines)):
            lnorm, _ = _norm_map(lines[j])
            if not lnorm:
                continue
            pos = lnorm.find(hnorm)
            if pos < 0:
                continue
            if lnorm == hnorm:
                cand = ("exact", j)
                break
            if pos == 0:
                residue = len(lnorm) - len(hnorm)
                _, idxs = _norm_map(lines[j])
                span_end = idxs[len(hnorm) - 1] + 1
                tail = lines[j][span_end:]
                tail_ch = tail[:1]
                tail_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", tail))
                if cand is not None:
                    continue
                if residue <= MATCH_RESIDUE and not tail_has_cjk:
                    cand = ("prefix", j)  # 行首标题 + 少量图标残差 → 替换整行
                elif tail_ch in " \t":
                    cand = ("split", j)  # 行首标题 + 空格 + 正文 → 原位拆分
                # 残差为 CJK 且无空格边界：正文句子恰好以标题词开头，不动
            elif pos + len(hnorm) == len(lnorm) and cand is None:
                cand = ("split", j)  # 行尾标题（被缝进上一行）→ 拆分保留前缀
        if cand is None:
            continue
        action, j = cand
        if action in ("exact", "prefix"):
            lines[j] = f"{marker} {text}"
            mi = j + 1
        else:  # 拆分：保留标题前的正文
            lnorm, idxs = _norm_map(lines[j])
            pos = lnorm.find(hnorm)
            raw_start = idxs[pos]
            prefix = lines[j][:raw_start].rstrip()
            head_line = f"{marker} {text}"
            lines[j : j + 1] = ([prefix] if prefix else []) + [head_line]
            mi = j + 2 if prefix else j + 1
        matched += 1
    return "\n".join(lines), matched, len(heads)


def strip_icon_bullets(text: str) -> str:
    """清理图标字体伪影。

    原 PDF 的列表符号是图标，无有效 ToUnicode 映射，文本层残留字符码：
    Webdings 'Q'（渲染为●圆点）、STXihei '█'（渲染为■方块）。
    仅在行首/空白后出现且后接空格，全书无以 "Q " 开头或孤立 "Q " 的正常内容，
    可安全移除。
    """
    # 行首（可带缩进、#标题、>引用 前缀），保留列表标记：
    text = re.sub(r"^[ \t]*((?:#+[ \t]*|>[ \t]*)?)-[ \t]*[Q█][ \t]+", r"\1- ", text, flags=re.M)
    text = re.sub(r"^[ \t]*((?:#+[ \t]*|>[ \t]*)?)[Q█][ \t]+", r"\1", text, flags=re.M)
    # 行中残留（pymupdf4llm 把同一行的多个图标项合并成一行）：删除 " Q"/" █"
    text = re.sub(r"(?<=\S)[ \t]+[Q█](?=[ \t]|$)", "", text)
    return text


def extract_chapter_map(heading_index: dict) -> dict[int, str]:
    """从主目录页解析 章号→章名 映射（值保留 "N 章名" 完整形态）。

    主目录页（前 12 页内）以 12pt 绿条样式列出全部 "N 章名" 行
    （1 安全 / 2 仪表组 / ... / 8 车辆规格）。值保留编号（"1 安全"）与
    节标题 "1-3 儿童保护装置" 的编号风格一致，便于原书溯源。
    找到含 ≥5 个编号条目的页即认定为主目录页。
    """
    for pg in sorted(heading_index):
        if pg > 12:
            break
        entries: dict[int, str] = {}
        for _y, text, marker in heading_index[pg]:
            m = re.match(r"^(\d+)\s+(\S.*)$", text.strip())
            if m and marker == "####":
                entries[int(m.group(1))] = f"{m.group(1)} {m.group(2).strip()}"
        if len(entries) >= 5:
            return entries
    return {}


def slim_chunks(md_chunks: list[dict]) -> list[dict]:
    """pymupdf4llm page_chunks 输出 → [{page_number, text}]。

    注意：页号在 chunk["metadata"]["page_number"]（顶层没有），单页子集模式为 None，
    此处退回顺序号兜底。
    """
    return [
        {
            "page_number": (c.get("metadata") or {}).get("page_number") or i,
            "text": c["text"],
        }
        for i, c in enumerate(md_chunks, 1)
    ]


# ---- 目录页判定（按"目录条目形态"而非孤立点线）----
# 页眉名黑名单：主目录 / 图片索引 / 字母索引
NOISE_SECTIONS = {"目录", "图片索引", "字母索引"}
# 目录条目的点线：连续 ≥4 个点，或 ≥2 个中文省略号（2~3 个点与正文省略号/
# 小数点/版本号不可分，故取下限 4）
DOT_LEADER_RUN_RE = re.compile(r"\.{4,}|\u2026{2,}")
TOC_LABEL_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]")  # 点线前 10 字符内须有中英文标签
TOC_DIGIT_RE = re.compile(r"\d")                     # 点线后 10 字符内须有页码
TOC_CONTEXT_WINDOW = 10
TOC_ENTRY_STRONG = 3      # 条目数 ≥3 直接判整页目录
TOC_ENTRY_MIN = 2         # 条目数 ≥2 才可能是目录页（恰好 2 条时另加尾部校验）
TOC_TWO_ENTRY_TAIL = 100  # 恰好 2 条时：末条目之后的正文长度上限（字符）
# 首个目录页之前的前部材料（封面/版权/前言/图例说明）属指路层之前的花絮，
# 无检索价值，整段剔除。仅当首个目录页出现在前 20 页内才启用：若目录在文末
# （部分报告/汇编），该规则会把整本文档删光，故必须设上界。
TOC_TRUNCATE_MAX_PAGE = 20


def _toc_entry_matches(text: str) -> list[re.Match]:
    """提取符合"目录条目形态"的点线匹配：点线前有标签、点线后有页码。

    取的是条目完整形态（标题 … 页码），而非孤立点线：表单点线、价格表
    点线、装饰分隔点线缺少标签或页码，不计入，故跨文档不易误伤。
    """
    ms = []
    for m in DOT_LEADER_RUN_RE.finditer(text):
        before = text[max(0, m.start() - TOC_CONTEXT_WINDOW):m.start()]
        after = text[m.end():m.end() + TOC_CONTEXT_WINDOW]
        if TOC_LABEL_RE.search(before) and TOC_DIGIT_RE.search(after):
            ms.append(m)
    return ms


def is_toc_page(text: str) -> bool:
    """整页是否为目录页（按目录条目形态判定，用于剔除章扉页迷你目录）。

    ≥3 条目录条目 → 直接判定为目录页。
    恰好 2 条 → 追加尾部校验：末条目之后若还有超过 100 字符的正文，说明
    这两条只是正文中凑巧两点线（如表单填写栏），不是整页目录，保留该页；
    真目录若只有 2 条，条目之后不会再有成段内容。

    计数单位是"点线出现次数"而非"行"：pymupdf4llm 会把相邻多个条目合并
    进同一渲染行（第 8 章扉页 6 个条目合成 1 行），按行计数会漏检；反向
    地一个条目被 <br> 拆成两行则会高估，故行数不是可靠代理。
    """
    ms = _toc_entry_matches(text)
    if len(ms) >= TOC_ENTRY_STRONG:
        return True
    if len(ms) >= TOC_ENTRY_MIN:
        return len(text[ms[-1].end():]) <= TOC_TWO_ENTRY_TAIL
    return False


def detect_structure(doc: pymupdf.Document):
    """样式签名探测：返回 (逐页章节标题, 标题索引, 章号→章名映射)。"""
    titles = extract_section_titles(doc)
    heading_index = build_heading_index(doc)
    chapter_map = extract_chapter_map(heading_index)
    return titles, heading_index, chapter_map


def _section_already_headed(body: str, title: str) -> bool:
    """章扉页保护：页面首个非空行若是与节标题同文的 h1 标题则不重复注入。

    章扉页的页眉带节标题常与 18pt 章标题同文（如"1 安全"），直接注入会
    产生 h1/h2 同名，面包屑出现"1 安全 > 1 安全"。
    """
    target = norm_text(title)
    if not target:
        return True
    for line in body.split("\n")[:3]:  # 章扉页大标题位于页首
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return norm_text(stripped.lstrip("#")) == target
        break
    return False


def clean_pages(
    md_chunks: list[dict],
    titles: list[str],
    heading_index: dict,
    doc: pymupdf.Document | None = None,
    chapter_map: dict[int, str] | None = None,
):
    """逐页清洗 + 标题注入 + 剔除目录页与其前部材料，返回 (记录, 统计)。

    平台（rag_platform）与 CLI 共用此函数，保证清洗逻辑只有一份实现。
    doc 传入时额外做表格结构重建（用 PDF 单元格底纹 + 文字坐标）；不传则
    保持旧行为（表格沿用 pymupdf4llm 的原始输出），兼容旧调用方。
    """
    # 首轮扫描：定位第一个目录页（页眉名命中 or 目录条目形态命中）
    first_toc = next(
        (
            c["page_number"]
            for c, t in zip(md_chunks, titles)
            if t in NOISE_SECTIONS or is_toc_page(c["text"])
        ),
        None,
    )
    # 目录页之前的内容（封面/版权/前言/图例说明）整段剔除；首个目录页出现在
    # 20 页之后时视为非书籍结构（目录可能在文末），不启用，避免删光全文。
    truncate_below = (
        first_toc
        if first_toc is not None and first_toc <= TOC_TRUNCATE_MAX_PAGE
        else None
    )

    records: list[dict] = []
    table_registry = []
    last_table = None
    n_noise = n_front = n_injected = n_detected = n_box_marked = n_box_total = 0
    last_section = ""
    last_chapter = ""
    n_h2_injected = 0
    for chunk, title in zip(md_chunks, titles):
        page_no = chunk["page_number"]
        if truncate_below is not None and page_no < truncate_below:
            n_front += 1
            continue
        if title in NOISE_SECTIONS:
            n_noise += 1
            continue
        if is_toc_page(chunk["text"]):
            n_noise += 1
            continue
        page_text = chunk["text"]
        if doc is not None:
            # 表格重建：pymupdf4llm 的 | 块 → PDF 单元格网格 → @@T{id}@@ 占位；
            # 跨页续表在这里把后续页的行并入上一页的 Table 对象
            page_text, last_table = replace_tables_in_page(
                doc[page_no - 1], page_no, page_text, table_registry, last_table,
                heading_index.get(page_no, []),
            )
        # 先清洗伪影（~~/<u>/反引号/CJK 空格），使粘连行中的标题文本连续可匹配，
        # 再做标题注入，最后清理图标残留
        cleaned = clean_markdown_artifacts(page_text)
        injected, matched, total = inject_headings(cleaned, heading_index.get(page_no, []))
        n_injected += matched
        n_detected += total
        # 提示框几何检测：框内行包 ">" 引用块（分块层确定性定界的依据）
        if doc is not None:
            regions = box_regions.detect_box_regions(doc[page_no - 1])
            injected, box_marked, box_total = box_regions.mark_box_lines(injected, regions)
            n_box_marked += box_marked
            n_box_total += box_total
        body = strip_icon_bullets(injected)
        if not body:
            continue
        # h2 节级注入：页眉带章节标题（前向填充）在切换页首注入 ## 行，
        # 使 fragment/面包屑携带 章 > 节 完整层级（与评测集 schema 对齐）。
        # h1 章级由主目录的 章号→章名 映射重建：章扉页因迷你目录常被整页
        # 剔除，扉页上的 18pt # 章名随之丢失，故按节标题章号查表补链。
        if title and title != last_section and not _section_already_headed(body, title):
            prefix_lines = []
            chap_match = re.match(r"^(\d+)(?=[-－\s])", title)
            chapter_name = (chapter_map or {}).get(int(chap_match.group(1))) if chap_match else None
            if chapter_name and chapter_name != last_chapter and chapter_name != title:
                prefix_lines.append(f"# {chapter_name}")
                last_chapter = chapter_name
            prefix_lines.append(f"## {title}")
            body = "\n".join(prefix_lines) + f"\n{body}"
            n_h2_injected += 1
        if title:
            last_section = title
        records.append({"page": page_no, "section": title, "text": body})

    # 全书处理完再替换占位符：跨页表的后续行此时已并入同一 Table 对象
    for rec in records:
        if table_registry:
            rec["text"] = resolve_placeholders(rec["text"], table_registry)

    stats = {
        "n_noise": n_noise,
        "n_front": n_front,
        "front_cutoff": truncate_below,
        "n_injected": n_injected,
        "n_detected": n_detected,
        "n_box_lines_marked": n_box_marked,
        "n_box_lines_total": n_box_total,
        "n_h2_injected": n_h2_injected,
        "n_chars": sum(len(r["text"]) for r in records),
        "n_records": len(records),
        "n_tables": len(table_registry),
        "n_table_merged": sum(t.n_merged for t in table_registry),
    }
    return records, stats
