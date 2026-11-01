"""嵌入客户端：通过 HTTP 调用独立 embedder 服务（OpenAI 兼容 /v1/embeddings）。

查询端与入库端必须使用同一模型、同一归一化配置，否则向量空间不一致；
该约束由 embedder 服务统一保证（api 与 embedder 共用同一镜像与配置）。
"""

import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# 单次请求编码一批文本，控制载荷与单请求耗时；读取超时需覆盖大批量编码
_CLIENT_BATCH = 128
_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化；normalize_embeddings 由服务端执行，与 Qdrant Cosine 距离配套。

    统一走 OpenAI 兼容协议：本地为 embedder 服务，云端为任意
    OpenAI 格式嵌入 API（改 .env 的 URL / KEY / MODEL 即可切换）。
    """
    if not texts:
        return []
    url = f"{settings.embedding_api_url.rstrip('/')}/embeddings"
    headers = (
        {"Authorization": f"Bearer {settings.embedding_api_key}"}
        if settings.embedding_api_key
        else {}
    )
    vectors: list[list[float]] = []
    with httpx.Client(timeout=_TIMEOUT) as client:
        for start in range(0, len(texts), _CLIENT_BATCH):
            batch = texts[start : start + _CLIENT_BATCH]
            response = client.post(
                url,
                json={"model": settings.embedding_model, "input": batch},
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()["data"]
            vectors.extend(
                item["embedding"]
                for item in sorted(data, key=lambda item: item["index"])
            )
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
