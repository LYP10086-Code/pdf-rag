"""BGE Reranker 客户端：调用 Cohere 兼容 /v1/rerank 服务。

只对混合检索召回的候选做精排：Top-N → 精排 → Top-K。
契约与云端 rerank 服务（Jina / SiliconFlow 等）一致，切换只改 .env
的 RERANKER_API_URL / RERANKER_API_KEY / RERANKER_MODEL。
服务异常时记录告警并进入短暂冷却期，冷却期内跳过重排，检索仍可用
（外部模型属真实边界）；重排只在查询期生效，切换服务不影响已建索引。
"""

import logging
import time

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# 连接失败快速返回（不拖慢检索）；单次精排可能数秒，读取超时放宽
_TIMEOUT = httpx.Timeout(120.0, connect=2.0)
_COOLDOWN_SECONDS = 60.0

_disabled_until = 0.0


def rerank(query: str, candidates: list[dict], text_key: str = "text") -> list[dict]:
    global _disabled_until
    # 开关判定在 retriever 层完成（未显式传参时才看全局默认）；
    # 显式调用即代表请求方要求重排，这里不再重复检查全局开关
    if not candidates:
        return candidates
    if time.monotonic() < _disabled_until:
        return candidates

    url = f"{settings.reranker_api_url.rstrip('/')}/rerank"
    headers = (
        {"Authorization": f"Bearer {settings.reranker_api_key}"}
        if settings.reranker_api_key
        else {}
    )
    try:
        response = httpx.post(
            url,
            json={
                "model": settings.reranker_model,
                "query": query,
                "documents": [candidate[text_key] for candidate in candidates],
                "return_documents": False,
            },
            headers=headers,
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json()["results"]
        if not results:
            raise ValueError("重排服务返回空结果")
        # 契约规定按 relevance_score 降序；个别服务受 top_n 影响可能截断，
        # 缺失的候选按原顺序追加在末尾
        scores = {
            item["index"]: float(item["relevance_score"]) for item in results
        }
        ranked = [
            {**candidate, "rerank_score": scores[index]}
            for index, candidate in enumerate(candidates)
            if index in scores
        ]
        missing = [
            candidate
            for index, candidate in enumerate(candidates)
            if index not in scores
        ]
        if missing:
            logger.warning("重排结果缺失 %s 条候选，按原顺序补在末尾", len(missing))
    except Exception as exc:  # noqa: BLE001  服务不可用不应拖垮检索
        _disabled_until = time.monotonic() + _COOLDOWN_SECONDS
        logger.warning("Reranker 服务不可用，%s 秒内跳过重排：%s", _COOLDOWN_SECONDS, exc)
        return candidates

    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked + missing
