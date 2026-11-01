"""结构化分块：标题 fragment（父块）+ 500/50 子块。

沿用现有验证过的三条约定：
1. 页尾标记 @@E{page}@@：页码跟随本页内容，避免标题切分器把页首内容归给上一片段
2. 标题 fragment 是父块（生成粒度），子块是检索粒度；两者通过 parent_key 关联
3. 表格按完整行分组、每片重复表头，不交给字符切分器硬切

解析结果没有逐页 markdown 时，用 PageLocator 在 PDF 页文本中定位块页码。
"""

import hashlib
import re
from dataclasses import dataclass, field

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core.config import settings
from app.ingestion import cleaner
from app.ingestion.parser import ParsedDocument

SENT_END = tuple("。！？；…" ")】》")

END_MARKER_RE = re.compile(r"@@E(\d+)@@")
END_TAIL_RE = re.compile(r"@@E?\d*@?$")
START_TAIL_RE = re.compile(r"^(?:E?\d*@@|@@E?)")

TABLE_LINE_RE = re.compile(r"^[ \t]*\|")
TABLE_NOTE_RE = re.compile(r"^[ \t]*(?:表格（跨第|表头说明：)")
TABLE_SEP_RE = re.compile(r"\|(?:\s*:?-{3,}:?\s*\|)+")
TABLE_PAGES_RE = re.compile(r"表格（跨第(\d+)至(\d+)页）")

BOX_TITLE_LINE_RE = re.compile(r"^\s*(警告|注意|温馨提示)(\s*[(（]\s*续\s*[)）])?\s*$")
# ">" 引用块：解析层提示框几何检测写入的确定性边界标记
BOX_MARK_RE = re.compile(r"^[ \t]*>[ \t]?")
BOX_CONT_RE = re.compile(r"^\s*(警告|注意|温馨提示)\s*[(（]\s*续\s*[)）]\s*$")

_NORM_KEEP = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")


@dataclass
class ChunkingResult:
    chunks: list[dict] = field(default_factory=list)
    parents: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def norm_text(text: str) -> str:
    return "".join(_NORM_KEEP.findall(text))


def merge_pages(pages: list[dict]) -> str:
    """同文档连续页合并：跨页断句直接拼接，页尾追加页码标记。"""
    out = ""
    prev_char = ""
    prev_page: int | None = None
    for page in pages:
        page_no = int(page["page_number"])
        if prev_page is not None:
            for missing in range(prev_page + 1, page_no):
                out += f"\n@@E{missing}@@"
        part = page["text"].strip()
        if part:
            starts_block = part.startswith(("#", "-", "|", "•", ">")) or re.match(
                r"^\d+[.、]", part
            )
            if not out:
                out = part
            elif prev_char in SENT_END or prev_char == "\n" or starts_block:
                out += "\n\n" + part
            else:
                out += part
            prev_char = part[-1]
        # 页尾标记跟随本页内容（跨页断句剥标记后保持连续）；仅当末行是标题时
        # 标记独立成行——否则会粘进 MarkdownHeaderTextSplitter 的标题元数据
        # （breadcrumb 出现 "@@E220@@" 污染）
        glue = "\n" if part.split("\n")[-1].lstrip().startswith("#") else ""
        out += f"{glue}@@E{page_no}@@"
        prev_page = page_no
    return out


def extract_pages(piece: str, cur_page: int | None) -> tuple[list[int], str, int | None]:
    """从页尾标记确定性计算页码，并剥离所有标记（定义见设计文档 5.5）。"""
    markers = [int(m) for m in END_MARKER_RE.findall(piece)]
    tail_stripped = END_MARKER_RE.sub("", piece.rstrip())
    truncated = bool(END_TAIL_RE.search(tail_stripped))

    if not markers:
        pages = [cur_page] if cur_page is not None else []
        new_cur = cur_page + 1 if (truncated and cur_page is not None) else cur_page
        clean = START_TAIL_RE.sub("", piece)
        clean = END_TAIL_RE.sub("", clean)
        clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
        return pages, clean, new_cur

    pages_set: set[int] = set()
    prev_end = 0
    for marker in markers:
        token = f"@@E{marker}@@"
        end = piece.index(token, prev_end)
        between = START_TAIL_RE.sub("", piece[prev_end:end])
        if between.strip():
            pages_set.add(marker)
        prev_end = end + len(token)
    after = END_TAIL_RE.sub("", piece[prev_end:])
    if after.strip():
        pages_set.add(markers[-1] + 1)
    new_cur = markers[-1] + 1 + (1 if truncated else 0)

    clean = END_MARKER_RE.sub("", piece)
    clean = START_TAIL_RE.sub("", clean)
    clean = END_TAIL_RE.sub("", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return sorted(pages_set), clean, new_cur


def clean_fragment_text(text: str) -> str:
    clean = END_MARKER_RE.sub("", text)
    clean = START_TAIL_RE.sub("", clean)
    clean = END_TAIL_RE.sub("", clean)
    clean = strip_box_marks(clean)
    return re.sub(r"\n{3,}", "\n\n", clean).strip()


def strip_box_marks(text: str) -> str:
    """剥掉行首 ">" 引用标记（解析层提示框边界），还原干净文本。"""
    return re.sub(r"(?m)^[ \t]*>[ \t]?", "", text)


def _unmarked_box_line(line: str) -> str:
    """供标题匹配：剥掉行首 ">" 引用标记后的行内容。"""
    return BOX_MARK_RE.sub("", line, count=1)


def build_breadcrumb(headings: dict, document_title: str = "") -> str:
    chain = [
        headings[key]
        for key in ("h1", "h2", "h3", "h4", "h5", "h6")
        if headings.get(key)
    ]
    parts = list(chain)
    if document_title and (not parts or parts[0] != document_title):
        parts = [document_title, *parts]
    return " > ".join(part for part in parts if part)


def strip_leading_headings(text: str) -> str:
    lines = text.split("\n")
    index = 0
    while index < len(lines) and lines[index].lstrip().startswith("#"):
        index += 1
    return "\n".join(lines[index:]).strip()


def _pages_string(pages: list[int]) -> str:
    if not pages:
        return ""
    return str(pages[0]) if pages[0] == pages[-1] else f"{pages[0]}-{pages[-1]}"


class PageLocator:
    """MinerU 无逐页 markdown 时，用页文本定位块的页码范围。"""

    def __init__(self, page_texts: list[str]) -> None:
        self._pages = [norm_text(text) for text in page_texts]

    def locate(self, text: str, hint: int | None = None) -> list[int]:
        target = norm_text(text)
        if not target or not self._pages:
            return [hint] if hint else []

        start = None
        for size in (40, 24, 16, 12):
            probe = target[:size]
            start = self._find(probe, hint)
            if start is not None:
                break
        if start is None:
            return [hint] if hint else []

        end = self._find(target[-40:], start)
        if end is None:
            end = start
        return list(range(start + 1, max(start, end) + 2))

    def _find(self, probe: str, hint: int | None) -> int | None:
        total = len(self._pages)
        begin = max(0, (hint or 1) - 1)
        order = list(range(begin, total)) + list(range(0, begin))
        for index in order:
            if probe and probe in self._pages[index]:
                return index
        return None


def split_table_markdown(block: str, chunk_size: int) -> list[str]:
    """长表按完整行分组，每片重复表头与表前说明行。"""
    lines = block.split("\n")
    note_lines = [line for line in lines if not TABLE_LINE_RE.match(line) and line.strip()]
    table_lines = [line for line in lines if TABLE_LINE_RE.match(line)]
    if not table_lines:
        return [block]

    tail_marker = ""
    cleaned = []
    for line in table_lines:
        found = END_MARKER_RE.findall(line)
        if found:
            tail_marker = f"@@E{found[-1]}@@"
        cleaned.append(END_MARKER_RE.sub("", line))
    table_lines = cleaned

    header = table_lines[0]
    if len(table_lines) >= 2 and TABLE_SEP_RE.fullmatch(table_lines[1].strip()):
        separator = table_lines[1]
        data_rows = table_lines[2:]
    else:
        columns = max(1, header.count("|") - 1)
        separator = "| " + " | ".join("---" for _ in range(columns)) + " |"
        data_rows = table_lines[1:]

    base_len = sum(len(line) + 1 for line in note_lines + [header, separator])
    pieces: list[str] = []
    current: list[str] = []
    current_len = base_len
    for row in data_rows:
        row_len = len(row) + 1
        if current and current_len + row_len > chunk_size:
            pieces.append("\n".join(note_lines + [header, separator] + current))
            current = []
            current_len = base_len
        current.append(row)
        current_len += row_len
    if current or not pieces:
        pieces.append("\n".join(note_lines + [header, separator] + current))

    if tail_marker:
        pieces[-1] = pieces[-1] + tail_marker
    return pieces


def split_fragment_text(text: str, size_splitter) -> list[tuple[str, str]]:
    """fragment → [(kind, piece)]：表格按行分组，其余走字符切分。"""
    lines = text.split("\n")
    segments: list[tuple[str, str]] = []
    buffer: list[str] = []
    index = 0
    while index < len(lines):
        if not TABLE_LINE_RE.match(lines[index]) or lines[index].count("|") < 3:
            buffer.append(lines[index])
            index += 1
            continue
        note: list[str] = []
        while buffer and TABLE_NOTE_RE.match(buffer[-1]):
            note.insert(0, buffer.pop())
        end = index
        while end < len(lines) and TABLE_LINE_RE.match(lines[end]):
            end += 1
        segments.append(("table", "\n".join(note + lines[index:end])))
        index = end
    if buffer:
        segments.append(("text", "\n".join(buffer)))

    pieces: list[tuple[str, str]] = []
    for kind, segment in segments:
        if not segment.strip():
            continue
        if kind == "table":
            pieces.extend(
                ("table", piece)
                for piece in split_table_markdown(segment, settings.chunk_size)
                if piece.strip()
            )
        else:
            pieces.extend(
                ("text", piece)
                for piece in size_splitter.split_text(segment)
                if piece.strip()
            )
    return pieces


def extract_box_supplements(
    frag_text: str,
    *,
    headings: dict,
    document_title: str,
    parent_key: str,
    page_for,
) -> list[dict]:
    """提示框补充块（multi-representation indexing）：警告类查询的独立入口。

    边界判定：
    - 标记模式（解析层几何检测已写入 ">"）：框 = 连续 ">" 行，遇到首个非引用
      非空行或下一个框标题即结束——确定性，正文不可能被吸入；
    - 回退模式（无任何 ">" 标记，如检测关闭或未命中）：沿用旧启发式——扫描到
      下一个 markdown 标题或下一个框标题为止。
    跨页续框拼接：标题带"（续）"是排版的显式续接信号，无条件并入 fragment 内
    最近的同类型框（警告/注意/温馨提示按基础词匹配），中间隔有的正文留在主块。

    与父块共享 parent_key，命中后按 key 取回完整 fragment。
    """
    lines = frag_text.split("\n")
    frag_norm = norm_text(END_MARKER_RE.sub("", frag_text))
    marked_mode = any(BOX_MARK_RE.match(line) for line in lines)
    box_starts = [
        i for i, line in enumerate(lines)
        if BOX_TITLE_LINE_RE.match(_unmarked_box_line(line).strip())
    ]
    if not box_starts:
        return []

    # 收集候选框：标题行 + 体部行（保留原始顺序）
    boxes: list[dict] = []
    for start in box_starts:
        title = _unmarked_box_line(lines[start]).strip()
        base_match = re.match(r"^\s*(警告|注意|温馨提示)", title)
        end = len(lines)
        for index in range(start + 1, len(lines)):
            stripped = lines[index].strip()
            if not stripped:
                continue  # 空行不终止
            if marked_mode:
                # 确定性边界：非 ">" 行，或下一个框标题行（框几何上互不重叠，
                # 标题行意味着上一框已结束——防止边界扫描穿过同页的后续框）
                if not BOX_MARK_RE.match(stripped) or BOX_TITLE_LINE_RE.match(
                    _unmarked_box_line(stripped).strip()
                ):
                    end = index
                    break
            elif stripped.startswith("#") or BOX_TITLE_LINE_RE.match(stripped):
                end = index
                break
        body = [_unmarked_box_line(line) for line in lines[start + 1 : end]]
        boxes.append(
            {
                "start": start,
                "end": end,
                "title": title,
                "base": base_match.group(1) if base_match else title,
                "body": body,
            }
        )

    # 跨页续框拼接：标题带（续）= 排版的显式续接信号，并入 fragment 内最近的
    # 同类型框（即使两半之间隔有正文——正文留在主块，不进框块）
    merged: list[dict] = []
    for box in boxes:
        if BOX_CONT_RE.match(box["title"]):
            target = next(
                (m for m in reversed(merged) if m["base"] == box["base"]), None
            )
            if target is not None:
                target["body"].extend(box["body"])
                target["end"] = box["end"]
                continue
        merged.append(box)

    out: list[dict] = []
    for box in merged:
        raw = "\n".join([box["title"], *box["body"]])
        if len(norm_text(END_MARKER_RE.sub("", raw))) >= 0.95 * len(frag_norm) > 0:
            continue  # 整个 fragment 就是一个框，避免与主块重复

        text = END_MARKER_RE.sub("", raw)
        text = START_TAIL_RE.sub("", text)
        text = END_TAIL_RE.sub("", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(norm_text(text)) < 4:
            continue

        start_offset = sum(len(line) + 1 for line in lines[: box["start"]])
        pages = page_for(start_offset)
        # 跨页续框拼接后，用框内最后一行的位置扩展页码范围（引用溯源显示 30-31）；
        # 不能用框结束位置——那已是框外第一行，落在次页时会把页码多标一页
        if box["body"] and pages:
            last_line_offset = sum(len(line) + 1 for line in lines[: box["end"] - 1])
            end_pages = page_for(last_line_offset)
            if end_pages and end_pages[-1] > pages[-1]:
                pages = list(range(pages[0], end_pages[-1] + 1))
        elif not pages:
            end_pages = page_for(sum(len(line) + 1 for line in lines[: box["end"]]))
            pages = pages or end_pages
        box_headings = {**headings, "h6": box["title"]}
        breadcrumb = build_breadcrumb(box_headings, document_title)
        title_line, _, body = text.partition("\n")

        pieces = [body]
        if len(text) > settings.supplement_max_chars:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.supplement_max_chars,
                chunk_overlap=settings.chunk_overlap,
                separators=["\n\n", "\n", "。", " ", ""],
            )
            pieces = splitter.split_text(body)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            out.append(
                {
                    "parent_key": parent_key,
                    "kind": "box",
                    "section": headings.get("h2") or headings.get("h1") or "",
                    "pages": _pages_string(pages),
                    "page_number": pages[0] if pages else None,
                    "breadcrumb": breadcrumb,
                    "text": f"{title_line}\n{piece}".strip(),
                    "embed_text": f"【{breadcrumb}】\n{piece}".strip(),
                }
            )
    return out


def build_chunks(
    parsed: ParsedDocument,
    document_id: str,
    document_title: str = "",
) -> ChunkingResult:
    headers = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=False
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", " ", ""],
    )

    locator: PageLocator | None = None
    dropped_pages = 0
    if parsed.page_aligned:
        total_pages = len(parsed.pages)
        records = cleaner.clean_page_texts(parsed.pages)
        dropped_pages = total_pages - len(records)
        if not records:
            raise ValueError("PDF 未解析出有效页面")
        merged = merge_pages(records)
        fragments = header_splitter.split_text(merged)
        cur_page: int | None = records[0]["page_number"]
        locate_hint: int | None = None
    else:
        markdown = cleaner.clean_markdown(parsed.markdown)
        if not markdown:
            raise ValueError("解析结果为空")
        fragments = header_splitter.split_text(markdown)
        locator = PageLocator(parsed.page_texts)
        cur_page = None
        locate_hint = 1

    chunks: list[dict] = []
    supplements: list[dict] = []
    parents: list[dict] = []
    stats = {
        "fragments": len(fragments),
        "tables": 0,
        "boxes": 0,
        "dropped_pages": dropped_pages,
    }

    def resolve_pages(piece: str) -> tuple[list[int], str]:
        nonlocal cur_page, locate_hint
        if parsed.page_aligned:
            pages, text, cur_page = extract_pages(piece, cur_page)
            return pages, text
        text = END_MARKER_RE.sub("", piece)
        text = START_TAIL_RE.sub("", text)
        text = END_TAIL_RE.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        probe_text = strip_leading_headings(text) or text
        pages = locator.locate(probe_text, locate_hint) if locator else []
        if pages:
            locate_hint = pages[-1]
        return pages, text

    for fragment_index, fragment in enumerate(fragments):
        headings = {
            key: value
            for key, value in fragment.metadata.items()
            if key in ("h1", "h2", "h3", "h4", "h5", "h6")
        }
        breadcrumb = build_breadcrumb(headings, document_title)
        section = headings.get("h2") or headings.get("h1") or ""
        parent_key = f"{document_id}:p{fragment_index}"

        if parsed.page_aligned:
            page_resolver = lambda pos, text=fragment.page_content: _marker_page(  # noqa: E731
                text, pos, cur_page
            )
        else:
            page_resolver = lambda pos, text=fragment.page_content: _locator_page(  # noqa: E731
                text, locator, locate_hint
            )

        supplements.extend(
            extract_box_supplements(
                fragment.page_content,
                headings=headings,
                document_title=document_title,
                parent_key=parent_key,
                page_for=page_resolver,
            )
        )

        fragment_pages: set[int] = set()
        for kind, piece in split_fragment_text(fragment.page_content, size_splitter):
            pages, text = resolve_pages(piece)
            text = strip_box_marks(text)
            if not text:
                continue
            pages_string = _pages_string(pages)
            if kind == "table":
                match = TABLE_PAGES_RE.search(text)
                if match:
                    pages_string = f"{match.group(1)}-{match.group(2)}"
                stats["tables"] += 1
            if pages:
                fragment_pages.update(pages)
            embed_text = (
                f"【{breadcrumb}】\n{strip_leading_headings(text)}" if breadcrumb else text
            )
            chunks.append(
                {
                    "parent_key": parent_key,
                    "kind": kind if kind == "table" else "main",
                    "section": section,
                    "pages": pages_string,
                    "page_number": pages[0] if pages else None,
                    "breadcrumb": breadcrumb,
                    "text": text,
                    "embed_text": embed_text,
                    "content_hash": hashlib.sha256(embed_text.encode("utf-8")).hexdigest(),
                }
            )

        parent_text = clean_fragment_text(fragment.page_content)
        if parent_text:
            pages_sorted = sorted(fragment_pages)
            parents.append(
                {
                    "parent_key": parent_key,
                    "section": section,
                    "pages": _pages_string(pages_sorted),
                    "page_number": pages_sorted[0] if pages_sorted else None,
                    "breadcrumb": breadcrumb,
                    "text": parent_text,
                }
            )

    stats["boxes"] = len(supplements)
    for supplement in supplements:
        supplement["content_hash"] = hashlib.sha256(
            supplement["embed_text"].encode("utf-8")
        ).hexdigest()
        chunks.append(supplement)

    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index

    stats.update({"chunks": len(chunks), "parents": len(parents)})
    return ChunkingResult(chunks=chunks, parents=parents, stats=stats)


def _marker_page(frag_text: str, pos: int, start_page: int | None) -> list[int]:
    markers = [(m.start(), int(m.group(1))) for m in END_MARKER_RE.finditer(frag_text)]
    for marker_pos, page in markers:
        if marker_pos > pos:
            return [page]
    if markers:
        return [markers[-1][1] + 1]
    return [start_page] if start_page else []


def _locator_page(frag_text: str, locator: PageLocator | None, hint: int | None) -> list[int]:
    if locator is None:
        return [hint] if hint else []
    return locator.locate(strip_leading_headings(frag_text) or frag_text, hint)
