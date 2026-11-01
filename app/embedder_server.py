"""独立嵌入服务：OpenAI 兼容 /v1/embeddings，由 compose 的 embedder 服务运行。

与 api 共用同一镜像，保证嵌入客户端与服务端的模型、依赖版本完全一致
（查询端与入库端必须使用同一模型、同一归一化配置，否则向量空间不一致）。
模型在启动时同步加载，加载完成前 uvicorn 不对外服务，/health 供 healthcheck 探活。
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
    from sentence_transformers import SentenceTransformer

    logger.info(
        "加载嵌入模型 %s（device=%s）", settings.embedding_model, settings.embedding_device
    )
    _model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    logger.info("嵌入模型加载完成")
    yield
    _model = None


app = FastAPI(title="pdf-rag-embedder", lifespan=lifespan)


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str] = Field(min_length=1)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.embedding_model,
        "device": settings.embedding_device,
    }


@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest) -> dict:
    if req.model and req.model != settings.embedding_model:
        raise HTTPException(
            status_code=400,
            detail=f"模型不一致：服务端 {settings.embedding_model}，请求 {req.model}",
        )
    if _model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    vectors = _encode(texts)
    return {
        "object": "list",
        "model": settings.embedding_model,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def _encode(texts: list[str]) -> list[list[float]]:
    """分批编码；normalize 与 Qdrant Cosine 距离配套，加锁避免 CPU 线程超订。"""
    vectors: list[list[float]] = []
    with _lock:
        for start in range(0, len(texts), settings.embedding_batch_size):
            batch = _model.encode(
                texts[start : start + settings.embedding_batch_size],
                batch_size=settings.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors.extend(v.tolist() for v in batch)
    return vectors
