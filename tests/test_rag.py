"""RAG 生成层与服务层纯函数测试（不调用真实 LLM）。"""

from app.rag.chain import build_citations, format_context
from app.rag.prompts import AGENT_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from app.services.document_service import sanitize_name


def _results() -> list[dict]:
    return [
        {
            "document_id": "d_1",
            "doc_name": "用户手册.pdf",
            "section": "1-2 安全气囊",
            "pages": "102-103",
            "page_number": 102,
            "breadcrumb": "1 安全 > 1-2 安全气囊 > 充电注意事项",
            "text": "充电时，启动车辆可使用空调。",
        },
        {
            "document_id": "d_2",
            "doc_name": "保养手册.pdf",
            "section": "6-2 定期保养",
            "pages": "233",
            "page_number": 233,
            "breadcrumb": "6 保养 > 6-2 定期保养",
            "text": "每 12 个月或 5000 公里检查一次。",
        },
    ]


def test_format_context_contains_traceable_metadata():
    context = format_context(_results())
    assert "[片段1]《用户手册.pdf》" in context
    assert "页码：p102-103" in context
    assert "[片段2]《保养手册.pdf》" in context
    assert "每 12 个月" in context


def test_build_citations_is_server_side_ordered():
    citations = build_citations(_results())
    assert [item["id"] for item in citations] == [1, 2]
    assert citations[0]["doc_name"] == "用户手册.pdf"
    assert citations[0]["page_number"] == 102
    assert citations[0]["snippet"].startswith("充电时")


def test_prompts_keep_safety_and_citation_rules():
    for prompt in (RAG_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT):
        assert "知识库中未找到相关内容" in prompt
        assert "原样引用" in prompt
        assert "装有时" in prompt
        assert "页码" in prompt


def test_sanitize_name_blocks_path_and_illegal_chars():
    assert sanitize_name("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_name("手册<>.pdf") == "手册__.pdf"
    assert sanitize_name("   ") == "document.pdf"
