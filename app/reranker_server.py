"""独立重排服务：Cohere 兼容 POST /v1/rerank，由 compose 的 reranker 服务运行。

与 api 共用同一镜像（仅启动命令不同），保证重排客户端/服务端
模型与依赖版本一致。采用 rerank 领域事实标准契约（Cohere /v1/rerank，
Jina / SiliconFlow 等云服务商同形），api 侧未来可无痛切换云端服务：
请求 {"model", "query", "documents"}，返回 results 按 relevance_score 降序。
模型在启动时同步加载。
"""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def _apply_hf_env() -> None:
    import os

    os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
    if settings.hf_hub_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model
    _apply_hf_env()
    from sentence_transformers import CrossEncoder

    logger.info(
        "加载重排模型 %s（device=%s）", settings.reranker_model, settings.reranker_device
    )
    _model = CrossEncoder(
        settings.reranker_model,
        device=settings.reranker_device,
        max_length=settings.reranker_max_length,
    )
    logger.info("重排模型加载完成")
    yield
    _model = None


app = FastAPI(title="pdf-rag-reranker", lifespan=lifespan)


class RerankRequest(BaseModel):
    model: str | None = None
    query: str = Field(min_length=1)
    # Cohere 契约允许纯字符串或 {"text": ...} 对象两种元素
    documents: list[str | dict]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.reranker_model,
        "device": settings.reranker_device,
    }


@app.post("/v1/rerank")
def rerank(req: RerankRequest) -> dict:
    if req.model and req.model != settings.reranker_model:
        raise HTTPException(
            status_code=400,
            detail=f"模型不一致：服务端 {settings.reranker_model}，请求 {req.model}",
        )
    if _model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    texts = [
        doc if isinstance(doc, str) else str(doc.get("text", ""))
        for doc in req.documents
    ]
    if not texts:
        return {"results": []}

    with _lock:
        scores = _model.predict(
            [(req.query, text) for text in texts],
            show_progress_bar=False,
        )
    results = [
        {"index": index, "relevance_score": float(score)}
        for index, score in enumerate(scores)
    ]
    results.sort(key=lambda item: item["relevance_score"], reverse=True)
    return {"results": results}
