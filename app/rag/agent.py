"""工具调用 Agent（LangChain 1.x `create_agent`）。

把检索封装成 search_knowledge_base 工具；Agent 可自行改写查询并决定是否再次检索，
适合模糊问题。限制递归次数，避免延迟不可控；检索仍复用同一份 retriever。
"""

import logging

from langchain.agents import create_agent
from langchain_core.tools import Tool

from app.core.config import settings
from app.core.database import session_scope
from app.rag.chain import build_llm, format_context
from app.rag.prompts import AGENT_SYSTEM_PROMPT
from app.retrieval.retriever import search

logger = logging.getLogger(__name__)


def _extract_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def ask_agent(
    question: str,
    *,
    user_id: str,
    knowledge_base_id: str,
    document_ids: list[str] | None = None,
    k: int | None = None,
) -> dict:
    captured: dict[str, list[dict]] = {"results": []}

    def search_knowledge_base(query: str) -> str:
        with session_scope() as db:
            payload = search(
                db,
                query,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                document_ids=document_ids,
                k=k,
            )
        results = payload["results"]
        captured["results"] = results
        if not results:
            return "未检索到相关内容。"
        return format_context(results)

    tool = Tool(
        name="search_knowledge_base",
        func=search_knowledge_base,
        description=(
            "在本地 PDF 知识库中检索与问题相关的资料。"
            "输入是完整的中文问题或关键词，返回带片段编号、文档名、章节和页码的资料。"
        ),
    )
    agent = create_agent(
        model=build_llm(),
        tools=[tool],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": settings.agent_max_iterations * 3 + 2},
    )
    messages = result["messages"]
    answer = _extract_text(messages[-1]) if messages else ""
    return {
        "answer": answer,
        "results": captured["results"],
        "steps": max(0, len(messages) - 2),
    }
