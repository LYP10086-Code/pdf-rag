"""BGE-M3 嵌入（CPU，进程内单例）。

查询端与入库端必须使用同一模型、同一归一化配置，否则向量空间不一致。
SentenceTransformer 延迟导入，避免仅做检索参数计算时也拉起 torch。
"""

import logging
import os
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)


def apply_hf_env() -> None:
    os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
    if settings.hf_hub_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


@lru_cache
def get_embedder():
    apply_hf_env()
    from sentence_transformers import SentenceTransformer

    logger.info(
        "加载嵌入模型 %s（device=%s）", settings.embedding_model, settings.embedding_device
    )
    return SentenceTransformer(settings.embedding_model, device=settings.embedding_device)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化；normalize_embeddings 与 Qdrant Cosine 距离配套。"""
    if not texts:
        return []
    vectors = get_embedder().encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
