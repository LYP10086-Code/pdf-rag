"""Qdrant 访问层（Docker/REST 部署）。

- collection `pdf_knowledge`：单个 dense 向量（1024 维，Cosine）
- payload 携带 user_id / knowledge_base_id / document_id 等定位信息，供检索时过滤
- 正文不写 payload（唯一权威在 PostgreSQL），避免双份数据不一致
"""

import logging
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.core.config import settings

logger = logging.getLogger(__name__)

# 固定命名空间：同一 (document_id, chunk_index) 每次得到相同 point id（幂等）
_NAMESPACE = uuid.UUID("0f3e4b8b-9d0a-4f6b-8c2e-000000000001")


@lru_cache
def get_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )


def ensure_collection() -> None:
    client = get_client()
    if client.collection_exists(settings.qdrant_collection):
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qm.VectorParams(
            size=settings.qdrant_vector_size,
            distance=qm.Distance.COSINE,
        ),
    )
    logger.info("创建 Qdrant collection：%s", settings.qdrant_collection)


def point_id(document_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{document_id}:{chunk_index}"))


def upsert_chunks(
    *,
    document_id: str,
    filename: str,
    user_id: str,
    knowledge_base_id: str,
    chunks: list[dict],
    vectors: list[list[float]],
) -> int:
    """按 chunk_index 写入向量与 payload；分批提交，避免单请求过大。"""
    ensure_collection()
    points = [
        qm.PointStruct(
            id=point_id(document_id, chunk["chunk_index"]),
            vector=vector,
            payload={
                "user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
                "document_id": document_id,
                "document_name": filename,
                "chunk_id": chunk["chunk_index"],
                "parent_key": chunk["parent_key"],
                "kind": chunk["kind"],
                "section": chunk["section"],
                "pages": chunk["pages"],
                "page_number": chunk["page_number"],
                "breadcrumb": chunk["breadcrumb"],
                "content_hash": chunk["content_hash"],
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client = get_client()
    for start in range(0, len(points), 128):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[start : start + 128],
        )
    return len(points)


def delete_document(document_id: str) -> None:
    ensure_collection()
    get_client().delete(
        collection_name=settings.qdrant_collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
            )
        ),
    )


def search_dense(
    *,
    query_vector: list[float],
    user_id: str,
    knowledge_base_id: str,
    document_ids: list[str] | None = None,
    limit: int = 20,
) -> list[dict]:
    """向量检索；权限与文档范围在 Qdrant 侧过滤（不是召回后再筛）。"""
    ensure_collection()
    must = [
        qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
        qm.FieldCondition(
            key="knowledge_base_id", match=qm.MatchValue(value=knowledge_base_id)
        ),
    ]
    if document_ids:
        must.append(
            qm.FieldCondition(key="document_id", match=qm.MatchAny(any=list(document_ids)))
        )
    points = get_client().query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=limit,
        query_filter=qm.Filter(must=must),
    ).points
    return [
        {
            "document_id": p.payload["document_id"],
            "chunk_index": int(p.payload["chunk_id"]),
            "score": float(p.score),
        }
        for p in points
    ]


def count() -> int:
    ensure_collection()
    return get_client().count(collection_name=settings.qdrant_collection).count


def close() -> None:
    try:
        get_client().close()
    finally:
        get_client.cache_clear()
