"""Markdown 清洗。

目标：去除 PDF 解析器产生的样式伪影，保留内容层原文。
- 删除线/高亮/下划线标签误判
- 反引号包裹引号
- CJK 字符间空格（PDF 字距导致）
- 图标字体残留（Q / █）
- 目录/索引页（点线条目形态，检索污染源）

不重写标题：结构由自研解析注入的 markdown 标题承载。
"""

import re

CJK = r"[\u4e00-\u9fff/]"

# 目录条目形态：点线（≥4 个点或 ≥2 个省略号）+ 前后标签/页码
DOT_LEADER_RUN_RE = re.compile(r"\.{4,}|\u2026{2,}")
TOC_LABEL_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]")
TOC_DIGIT_RE = re.compile(r"\d")
TOC_CONTEXT_WINDOW = 10
TOC_ENTRY_STRONG = 3
TOC_ENTRY_MIN = 2
TOC_TWO_ENTRY_TAIL = 100


def clean_markdown_artifacts(text: str) -> str:
    text = text.replace("~~", "")  # 中文加粗被误判为删除线
    text = re.sub(r"</?(?:mark|u)>", "", text)
    text = re.sub(r"`+([“”])`+", r"\1", text)
    text = re.sub(r"`+([“”])", r"\1", text)
    text = re.sub(r"([“”])`+", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(rf"(?<={CJK})[ \t]+(?={CJK})", "", text)
    return text.strip()


def strip_icon_bullets(text: str) -> str:
    """清理图标字体残留：Webdings 'Q' 与 STXihei '█'。"""
    text = re.sub(
        r"^[ \t]*((?:#+[ \t]*|>[ \t]*)?)-[ \t]*[Q█][ \t]+", r"\1- ", text, flags=re.M
    )
    text = re.sub(r"^[ \t]*((?:#+[ \t]*|>[ \t]*)?)[Q█][ \t]+", r"\1", text, flags=re.M)
    text = re.sub(r"(?<=\S)[ \t]+[Q█](?=[ \t]|$)", "", text)
    return text


def clean_markdown(text: str) -> str:
    return strip_icon_bullets(clean_markdown_artifacts(text))


def _toc_entry_matches(text: str) -> list[re.Match]:
    matches = []
    for match in DOT_LEADER_RUN_RE.finditer(text):
        before = text[max(0, match.start() - TOC_CONTEXT_WINDOW) : match.start()]
        after = text[match.end() : match.end() + TOC_CONTEXT_WINDOW]
        if TOC_LABEL_RE.search(before) and TOC_DIGIT_RE.search(after):
            matches.append(match)
    return matches


def is_toc_page(text: str) -> bool:
    """整页目录/索引判定：≥3 条点线条目，或恰好 2 条且条目后无成段正文。"""
    matches = _toc_entry_matches(text)
    if len(matches) >= TOC_ENTRY_STRONG:
        return True
    if len(matches) >= TOC_ENTRY_MIN:
        return len(text[matches[-1].end() :]) <= TOC_TWO_ENTRY_TAIL
    return False


def clean_page_texts(pages: list[dict]) -> list[dict]:
    """逐页清洗；空页与目录页丢弃，但保留原页码。"""
    cleaned = []
    for page in pages:
        text = clean_markdown(page["text"])
        if text and not is_toc_page(text):
            cleaned.append({"page_number": page["page_number"], "text": text})
    return cleaned
