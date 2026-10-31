"""检索与分块纯函数测试（不依赖 PostgreSQL / Qdrant / 模型）。"""

from app.ingestion.chunker import (
    END_MARKER_RE,
    PageLocator,
    build_breadcrumb,
    build_chunks,
    extract_pages,
    merge_pages,
    split_table_markdown,
)
from app.ingestion.parser import ParsedDocument
from app.retrieval.hybrid import rrf_fuse, tokenize


def test_tokenize_cjk_bigram_and_latin():
    tokens = tokenize("座椅侧 AIRBAG 2024")
    assert "座椅" in tokens
    assert "椅侧" in tokens
    assert "airbag" in tokens
    assert "2024" in tokens


def test_rrf_prefers_consensus_candidate():
    bm25_keys = [("d1", 1), ("d1", 2)]
    dense_keys = [("d1", 2), ("d1", 3)]
    fused = rrf_fuse(bm25_keys, dense_keys, bm25_weight=0.2, rrf_k=60)
    assert fused["order"][0] == ("d1", 2)
    assert fused["score"][("d1", 2)] > fused["score"][("d1", 1)]


def test_extract_pages_marks_and_cleans_text():
    pages, text, cur_page = extract_pages("前半@@E1@@后半@@E2@@", 1)
    assert pages == [1, 2]
    assert "@@" not in text
    assert text.startswith("前半")
    assert cur_page == 3


def test_merge_pages_joins_cross_page_sentence():
    merged = merge_pages(
        [
            {"page_number": 1, "text": "为获得座椅侧安全气囊的最"},
            {"page_number": 2, "text": "佳保护，乘员必须正确佩戴安全带。"},
        ]
    )
    assert "为获得座椅侧安全气囊的最佳保护" in END_MARKER_RE.sub("", merged)
    assert "@@E1@@" in merged and "@@E2@@" in merged


def test_split_table_repeats_header():
    table = "|项目|数值|\n|---|---|\n|座椅|4|\n|气囊|6|\n"
    pieces = split_table_markdown(table, chunk_size=20)
    assert len(pieces) >= 2
    assert all("|项目|数值|" in piece for piece in pieces)


def test_page_locator_finds_page_range():
    locator = PageLocator(["第一页：安全气囊简介", "第二页：儿童保护装置", "第三页：保养周期"])
    pages = locator.locate("儿童保护装置")
    assert pages == [2]


def test_build_breadcrumb_includes_document_title():
    breadcrumb = build_breadcrumb({"h1": "1 安全", "h2": "1-2 安全气囊"}, "测试手册")
    assert breadcrumb == "测试手册 > 1 安全 > 1-2 安全气囊"


def test_build_chunks_page_markers_and_box_supplement():
    parsed = ParsedDocument(
        markdown="",
        page_texts=[],
        pages=[
            {
                "page_number": 1,
                "text": "# 测试手册\n\n## 1 安全\n\n### 安全带\n\n为获得座椅侧安全气囊的最",
            },
            {
                "page_number": 2,
                "text": "佳保护。\n\n警告\n\n严禁儿童乘坐在前排座椅位置。\n\n继续检查安全带卡扣。",
            },
        ],
        page_count=2,
        parser="custom",
        page_aligned=True,
    )
    result = build_chunks(parsed, document_id="d_test", document_title="测试手册")

    assert result.chunks, "至少应生成一个子块"
    assert result.parents, "应生成父块"
    merged_text = "".join(chunk["text"] for chunk in result.chunks)
    assert "最佳保护" in merged_text
    assert any(chunk["kind"] == "box" for chunk in result.chunks)
    assert any(
        "测试手册 > 1 安全 > 安全带" in (chunk["breadcrumb"] or "")
        for chunk in result.chunks
    )
    assert all(
        chunk["embed_text"].startswith("【")
        for chunk in result.chunks
        if chunk["breadcrumb"]
    )
    assert all(chunk["chunk_index"] == index for index, chunk in enumerate(result.chunks))
    assert all("@@" not in chunk["text"] for chunk in result.chunks), "页码标记必须被完全剥离"


def test_build_chunks_locates_pages_without_page_markers():
    parsed = ParsedDocument(
        markdown="# 手册\n\n## 第一章\n\n第一页正文内容。\n\n## 第二章\n\n第二页正文内容。",
        page_texts=["第一页正文内容。", "第二页正文内容。"],
        pages=[],
        page_count=2,
        parser="custom",
        page_aligned=False,
    )
    result = build_chunks(parsed, document_id="d_mineru", document_title="手册")
    assert result.chunks
    located = [chunk for chunk in result.chunks if chunk["page_number"]]
    assert located, "MinerU 路径应能定位到页码"
    assert any(chunk["page_number"] == 2 for chunk in located)
