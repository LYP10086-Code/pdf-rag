"""BGE Reranker（CPU，CrossEncoder）。

只对混合检索召回的候选做精排：Top-N → 精排 → Top-K。
模型加载失败或推理异常时记录告警并跳过重排，检索仍可用（外部模型属真实边界）。
"""

import logging
from functools import lru_cache

from app.core.config import settings
from app.core.embeddings import apply_hf_env

logger = logging.getLogger(__name__)
_FAILED = False


@lru_cache
def get_reranker():
    from sentence_transformers import CrossEncoder

    apply_hf_env()

    logger.info("加载重排模型 %s（device=%s）", settings.reranker_model, settings.reranker_device)
    return CrossEncoder(
        settings.reranker_model,
        device=settings.reranker_device,
        max_length=settings.reranker_max_length,
    )


def rerank(query: str, candidates: list[dict], text_key: str = "text") -> list[dict]:
    global _FAILED
    if not settings.reranker_enabled or not candidates or _FAILED:
        return candidates

    try:
        model = get_reranker()
        scores = model.predict(
            [(query, candidate[text_key]) for candidate in candidates],
            show_progress_bar=False,
        )
    except Exception as exc:  # noqa: BLE001  模型下载/显存问题不应拖垮检索
        _FAILED = True
        logger.warning("Reranker 不可用，跳过重排：%s", exc)
        return candidates

    ranked = []
    for candidate, score in zip(candidates, scores, strict=True):
        item = {**candidate, "rerank_score": float(score)}
        ranked.append(item)
    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked
