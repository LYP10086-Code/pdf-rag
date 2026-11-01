"""MCP + LangChain Agent loop demo。

把 pdf-rag 暴露的 MCP 工具（http://localhost:8080/mcp/）挂到
LangChain 1.x 的 create_agent 循环上，跑一轮
「解析车型 → list_documents 拿 document_id → search_knowledge_base 检索 → 生成答案」
的完整决策过程，并打印每一轮的工具调用与模型输出。

运行（WSL venv；langchain-mcp-adapters 是 demo 专用依赖，不在 requirements.txt 里）：
    pip install langchain-mcp-adapters
    python mcp_agent_demo.py
"""

import asyncio
import json

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

MCP_URL = "http://localhost:8080/mcp/"
LLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "Qwen/Qwen3-4B-AWQ"

QUESTION = "我的车是秦Plus EV 我车钥匙没电了怎么办"

SYSTEM_PROMPT = """你是汽车用户手册问答助手，通过 MCP 工具查询本地 PDF 知识库。
规则：
1. 涉及车辆功能、故障、操作的问题必须先用工具检索，不许凭记忆回答。
2. 需要按文档收窄范围时，先用 list_documents 把车型名解析成 document_id，
   再把它传入 search_knowledge_base 的 document_ids；不确定时也可以
   直接全库检索，把车型名写进问题里即可。
3. 最终回答 250 字以内，标注文档名与页码；资料不足就明说。"""


def print_message(message) -> None:
    """打印一条消息：AI 的决策/输出，或工具返回的摘要。"""
    if isinstance(message, AIMessage):
        for call in message.tool_calls or []:
            args = json.dumps(call["args"], ensure_ascii=False)
            print(f"  [AI 调用工具] {call['name']}  参数: {args}")
        if message.content:
            print(f"  [AI 输出] {message.content}")
    elif isinstance(message, ToolMessage):
        name = getattr(message, "name", "tool")
        text = message.content if isinstance(message.content, str) else str(message.content)
        preview = text[:400] + ("…(截断)" if len(text) > 400 else "")
        print(f"  [工具返回 {name}] {preview}")


async def main() -> None:
    client = MultiServerMCPClient(
        {"pdf-rag": {"url": MCP_URL, "transport": "streamable_http"}}
    )
    tools = await client.get_tools()
    print(f"已挂载 MCP 工具: {sorted(t.name for t in tools)}")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key="EMPTY",
        temperature=0.1,
        max_tokens=512,
        timeout=120,
        # Qwen3 默认开思考模式，关掉避免 think 块占用 4096 上下文预算
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

    print(f"问题: {QUESTION}")
    print("=" * 60)

    seen = 0
    messages = []
    # MCP 适配器生成的工具只支持异步调用，必须用 astream 而不是 stream
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": QUESTION}]},
        config={"recursion_limit": 10},
        stream_mode="values",
    ):
        messages = chunk["messages"]
        for message in messages[seen:]:
            print_message(message)
        seen = len(messages)

    print("=" * 60)
    print("最终答案:")
    print(messages[-1].content if messages else "<无输出>")


if __name__ == "__main__":
    asyncio.run(main())
