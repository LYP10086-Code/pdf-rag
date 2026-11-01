"""FastAPI 入口。

启动顺序：建 PostgreSQL 表 → 建 Qdrant collection → 恢复残留任务。
文档入库在进程内单线程队列执行，不阻塞问答接口。
"""

import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select, text

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.core import qdrant
from app.core.config import settings
from app.core.database import init_db, session_scope
from app.core.models import Document
from app.mcp_server import build_mcp_asgi, mcp as mcp_server
from app.services import document_service

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Windows 注册表可能把 .js 映射成 text/plain，导致浏览器拒绝执行 ES module
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")


# MCP 子应用：模块级构建（session_manager 只有在 streamable_http_app()
# 调用后才存在，宿主 lifespan 才能启动它）；端点 = 挂载前缀 /mcp/
mcp_asgi = build_mcp_asgi()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MCP 挂载后子应用 lifespan 不会执行，session manager 由宿主 lifespan 启动
    async with mcp_server.session_manager.run():
        settings.ensure_dirs()
        init_db()
        qdrant.ensure_collection()
        with session_scope() as db:
            stale = document_service.recover_stale_jobs(db)
        if stale:
            logger.info("启动时将 %d 个中断任务标记为 failed", stale)
        logger.info("pdf-rag 启动完成：%s", settings.app_version)
        yield
        document_service.shutdown()
        qdrant.close()


app = FastAPI(
    title="PDF-RAG 本地知识库",
    version=settings.app_version,
    description=(
        "PDF 上传 → 自研解析 → 分块 → BGE-M3 向量化 → Qdrant 检索 "
        "→ BGE 重排 → Qwen3 生成答案与引用。"
    ),
    lifespan=lifespan,
)
app.include_router(documents_router)
app.include_router(chat_router)

# MCP：检索以标准工具协议暴露给外部 Agent（streamable HTTP）
app.mount("/mcp", mcp_asgi, name="mcp")


@app.get("/api/health")
def health():
    db_ok = False
    documents_total = 0
    documents_ready = 0
    try:
        with session_scope() as db:
            db.execute(text("SELECT 1"))
            db_ok = True
            documents_total = db.scalar(select(func.count()).select_from(Document)) or 0
            documents_ready = (
                db.scalar(
                    select(func.count()).select_from(Document).where(Document.status == "ready")
                )
                or 0
            )
    except Exception as exc:  # noqa: BLE001  health 应报告状态而不是直接 500
        logger.warning("健康检查数据库失败：%s", exc)

    try:
        vector_points = qdrant.count()
        qdrant_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("健康检查 Qdrant 失败：%s", exc)
        vector_points = None
        qdrant_ok = False

    return {
        "status": "ok" if db_ok and qdrant_ok else "degraded",
        "database": {
            "ok": db_ok,
            "documents_total": documents_total,
            "documents_ready": documents_ready,
        },
        "vector_store": {
            "ok": qdrant_ok,
            "backend": "qdrant",
            "url": settings.qdrant_url,
            "collection": settings.qdrant_collection,
            "points": vector_points,
        },
        "models": {
            "llm": settings.llm_model,
            "llm_base_url": settings.llm_base_url,
            "embedding": settings.embedding_model,
            "embedding_device": settings.embedding_device,
            "reranker": settings.reranker_model if settings.reranker_enabled else None,
            "pdf_parser": settings.pdf_parser,
        },
        "storage": str(settings.storage_path),
    }


# 前端（Vite 构建产物）：必须最后挂载——挂载点 "/" 会吞掉之后注册的路由
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")
else:

    @app.get("/")
    def root():
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "note": "前端未构建：在 web/ 下执行 npm install && npm run build",
            "docs": "/docs",
            "health": "/api/health",
        }
