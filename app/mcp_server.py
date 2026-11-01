"""MCP 服务器：把文档列表与知识库检索暴露为 MCP 工具，供外部 Agent 调用。

- 传输：streamable HTTP，由 main.py 挂载到 /mcp，端点为 http://<host>:<port>/mcp/
- 检索复用与 /api/search、内部 Agent 工具同一份 retriever 实现，行为完全一致
- 质量参数（k、父块归并、重排）不进工具签名：由服务端 .env 统一决策，
  工具侧只传语义参数（query、可选 document_ids），管线升级对调用方透明
- document_ids 是精确过滤（Qdrant MatchAny / SQL IN）：Agent 需先用
  list_documents 把车型名解析成 document_id，解析不到就全库检索

挂载注意：子应用自带的 lifespan 不会执行，session_manager 必须由宿主
lifespan 启动（见 main.py）；且 session_manager 在 build_mcp_asgi() 调用
后才存在，因此构建必须发生在 lifespan 触碰它之前（模块级调用即可）。
"""

import logging

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette

from app.core.config import settings
from app.core.database import session_scope
from app.rag.chain import format_context
from app.retrieval.retriever import search
from app.services import document_service

logger = logging.getLogger(__name__)

mcp = MCPServer("pdf-rag")


@mcp.tool()
def list_documents(query: str | None = None) -> str:
    """列出知识库中的文档，把车型名或文件名解析成 document_id。

    可选 query 按文件名模糊筛选（如"秦PLUS"），不传则列出全部，最多 50 条。
    文件名匹配是纯子串（大小写不敏感）；带空格的写法（如"秦 PLUS"）未命中时
    会自动去掉空白按连续子串重查一次。
    每行格式：document_id｜文件名｜状态｜页数/分块数。
    只有状态 ready 的文档可被检索；需要收窄 search_knowledge_base 的
    检索范围时，把这里返回的 document_id 原样传入其 document_ids 参数。
    """
    with session_scope() as db:
        payload = document_service.list_documents(db, query=query, size=50)
        if not payload["items"] and query:
            # 兜底：Agent 常把车型名写成"秦 PLUS"这类带空格的形式，而文件名
            # ilike 是纯子串匹配；去掉空白后按连续子串重查一次
            compact = "".join(query.split())
            if compact and compact != query:
                payload = document_service.list_documents(db, query=compact, size=50)
    items = payload["items"]
    if not items:
        return "知识库中没有匹配的文档。"
    lines = []
    for item in items:
        pages = item.get("page_count")
        lines.append(
            f"{item['id']}｜{item['original_name']}｜状态 {item['status']}"
            f"｜{pages if pages is not None else '?'} 页 / {item['chunk_count']} 块"
        )
    header = f"共 {payload['total']} 个文档"
    if payload["total"] > len(items):
        header += f"（仅显示前 {len(items)} 条）"
    return header + "：\n" + "\n".join(lines)


@mcp.tool()
def search_knowledge_base(query: str, document_ids: list[str] | None = None) -> str:
    """在本地 PDF 知识库中检索与问题相关的资料片段。

    输入完整的中文问题或关键词；document_ids 可选（来自 list_documents），
    不传即全库检索——车型名直接写进问题里通常就能命中对应文档。
    返回带片段编号、文档名、章节、页码的资料块，片段编号可作为回答的引用依据；
    知识库中没有相关内容时会明确返回"未检索到相关内容"。
    """
    cleaned = query.strip()
    if not cleaned:
        return "查询为空，请输入具体的问题或关键词。"
    with session_scope() as db:
        payload = search(
            db,
            cleaned,
            user_id=settings.default_user_id,
            knowledge_base_id=settings.default_kb_id,
            document_ids=document_ids,
        )
    if not payload["results"]:
        return "未检索到相关内容。"
    return format_context(payload["results"])


def _transport_security() -> TransportSecuritySettings:
    """Host 允许列表防 DNS 重绑定；MCP_ALLOWED_HOSTS 留空时关闭校验（内网 demo 默认）。"""
    hosts = [host.strip() for host in settings.mcp_allowed_hosts.split(",") if host.strip()]
    if not hosts:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return TransportSecuritySettings(allowed_hosts=hosts)


def build_mcp_asgi() -> Starlette:
    """构建可挂载到 FastAPI 的 ASGI 子应用（端点路径 = 挂载前缀本身）。"""
    asgi = mcp.streamable_http_app(
        streamable_http_path="/",
        transport_security=_transport_security(),
    )
    logger.info("MCP 服务就绪：工具 list_documents / search_knowledge_base 暴露于 /mcp/")
    return asgi
