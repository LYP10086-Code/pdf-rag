"""混合检索：BM25（应用内）+ 向量（Qdrant）→ 加权 RRF。

沿用现有 78 题评测结论：
- BM25 用 CJK 二元组分词（Lucene CJKAnalyzer 口径），索引 embed_text（含面包屑）
- BM25 权重 0.2、向量 1.0、RRF k=60；该档位是所有指标均不劣于纯向量的配置
- BM25 索引按 (user, knowledge_base, document_ids) 缓存，入库/删除后失效
"""

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.models import Chunk, Document, DocumentStatus

logger = logging.getLogger(__name__)

_LATIN_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """CJK 二元组 + 英文数字词；不引入词典分词依赖。"""
    text = text.lower()
    tokens = _LATIN_RE.findall(text)
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


@dataclass
class BM25Index:
    bm25: BM25Okapi
    keys: list[tuple[str, int]]


_CACHE: dict[tuple, BM25Index] = {}


def invalidate_bm25_cache() -> None:
    _CACHE.clear()


def get_bm25_index(
    db: Session,
    *,
    user_id: str,
    knowledge_base_id: str,
    document_ids: list[str] | None = None,
) -> BM25Index | None:
    cache_key = (user_id, knowledge_base_id, tuple(sorted(document_ids or [])))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    stmt: Select = (
        select(Chunk.document_id, Chunk.chunk_index, Chunk.embed_text)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Document.user_id == user_id,
            Document.knowledge_base_id == knowledge_base_id,
            Document.status == DocumentStatus.READY,
        )
        .order_by(Chunk.document_id, Chunk.chunk_index)
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(list(document_ids)))

    rows = db.execute(stmt).all()
    if not rows:
        return None

    keys = [(row[0], row[1]) for row in rows]
    index = BM25Index(bm25=BM25Okapi([tokenize(row[2]) for row in rows]), keys=keys)
    _CACHE[cache_key] = index
    logger.info("构建 BM25 索引：%d 块", len(keys))
    return index


def bm25_candidates(
    index: BM25Index | None,
    query: str,
    depth: int,
) -> tuple[list[tuple[str, int]], dict[tuple[str, int], float]]:
    if index is None:
        return [], {}
    scores = index.bm25.get_scores(tokenize(query))
    order = sorted(range(len(index.keys)), key=lambda i: -scores[i])[:depth]
    keys = [index.keys[i] for i in order if scores[i] > 0]
    return keys, {index.keys[i]: float(scores[i]) for i in order}


def rrf_fuse(
    bm25_keys: list,
    dense_keys: list,
    bm25_weight: float | None = None,
    rrf_k: int | None = None,
) -> dict:
    """加权 RRF：w_bm25/(k+r_bm25) + 1/(k+r_dense)。

    并列按 (最小排名, 向量排名) 定序，避免排序稳定性让 BM25 系统性优先。
    """
    weight = settings.hybrid_bm25_weight if bm25_weight is None else bm25_weight
    k = settings.rrf_k if rrf_k is None else rrf_k

    score: dict = {}
    best: dict = {}
    for channel_weight, keys in ((weight, bm25_keys), (1.0, dense_keys)):
        if channel_weight == 0:
            continue
        for rank, key in enumerate(keys, 1):
            score[key] = score.get(key, 0.0) + channel_weight / (k + rank)
            best[key] = min(best.get(key, 10 ** 9), rank)

    dense_pos = {key: rank for rank, key in enumerate(dense_keys, 1)}
    order = sorted(
        score,
        key=lambda key: (-score[key], best[key], dense_pos.get(key, 10 ** 9)),
    )
    return {"score": score, "best": best, "order": order}
