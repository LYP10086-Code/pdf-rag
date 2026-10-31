"""RAG 生成链：检索结果 → 带引用的提示词 → vLLM/Qwen3 → 答案。

LCEL 实现沿用现有单轮 RAG 脚本的约束：资料编号、原样引用警告、
资料不足拒答，引用信息由服务端统一输出，模型不得自行编造页码。
"""

import logging
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.rag.prompts import HUMAN_PROMPT, RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

NONE_ANSWER = "知识库中未找到相关内容。"


@lru_cache
def build_llm() -> ChatOpenAI:
    logger.info("LLM：%s @ %s", settings.llm_model, settings.llm_base_url)
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=120,
        # Qwen3 默认开启思考模式；RAG 问答只需最终答案，避免 think 块占用生成预算
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def format_context(results: list[dict]) -> str:
    """把检索结果拼成带编号、文档名、章节、页码的资料块。"""
    blocks = []
    for index, result in enumerate(results, 1):
        location = result.get("breadcrumb") or result.get("section") or ""
        pages = result.get("pages") or "未知"
        blocks.append(
            f"[片段{index}]《{result.get('doc_name', '未知文档')}》 {location}｜页码：p{pages}\n"
            f"{result['text'].strip()}"
        )
    return "\n\n".join(blocks)


def build_citations(results: list[dict]) -> list[dict]:
    """服务端统一生成引用列表，供前端展示与答案校验。"""
    return [
        {
            "id": index,
            "document_id": result.get("document_id"),
            "doc_name": result.get("doc_name"),
            "section": result.get("section"),
            "pages": result.get("pages"),
            "page_number": result.get("page_number"),
            "breadcrumb": result.get("breadcrumb"),
            "snippet": result.get("text", "")[:160],
        }
        for index, result in enumerate(results, 1)
    ]


@lru_cache
def build_rag_chain():
    prompt = ChatPromptTemplate.from_messages(
        [("system", RAG_SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    return prompt | build_llm() | StrOutputParser()


def answer_with_context(question: str, context: str) -> str:
    return build_rag_chain().invoke({"question": question, "context": context})
