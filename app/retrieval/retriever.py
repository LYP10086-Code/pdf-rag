"""检索核心：向量 + BM25 → RRF → 重排 → 父块归并。

接口与 CLI、Agent 工具共用同一实现，避免“线上与评测两套逻辑”漂移。
- 权限与文档范围在 Qdrant / SQL 查询阶段过滤，不做召回后二次筛选
- merge_parents=False 返回融合排序的 child，用于对比原始命中片段
"""

import logging
import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core import qdrant
from app.core.config import settings
from app.core.embeddings import embed_query
from app.core.models import Chunk, Document, ParentChunk
from app.retrieval import reranker
from app.retrieval.hybrid import bm25_candidates, get_bm25_index, rrf_fuse

logger = logging.getLogger(__name__)


def _load_chunks(db: Session, keys: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    if not keys:
        return {}
    document_ids = {document_id for document_id, _ in keys}
    chunk_indices = {chunk_index for _, chunk_index in keys}
    rows = (
        db.execute(
            select(Chunk).where(
                Chunk.document_id.in_(document_ids),
                Chunk.chunk_index.in_(chunk_indices),
            )
        )
        .scalars()
        .all()
    )
    wanted = set(keys)
    return {
        (row.document_id, row.chunk_index): {
            "document_id": row.document_id,
            "chunk_index": row.chunk_index,
            "parent_key": row.parent_key,
            "kind": row.kind,
            "section": row.section,
            "pages": row.pages,
            "page_number": row.page_number,
            "breadcrumb": row.breadcrumb,
            "text": row.text,
        }
        for row in rows
        if (row.document_id, row.chunk_index) in wanted
    }


def _load_parents(
    db: Session, parent_keys: set[tuple[str, str]]
) -> dict[tuple[str, str], dict]:
    if not parent_keys:
        return {}
    document_ids = {document_id for document_id, _ in parent_keys}
    keys = {parent_key for _, parent_key in parent_keys}
    rows = (
        db.execute(
            select(ParentChunk).where(
                ParentChunk.document_id.in_(document_ids),
                ParentChunk.parent_key.in_(keys),
            )
        )
        .scalars()
        .all()
    )
    wanted = set(parent_keys)
    return {
        (row.document_id, row.parent_key): {
            "section": row.section,
            "pages": row.pages,
            "page_number": row.page_number,
            "breadcrumb": row.breadcrumb,
            "text": row.text,
        }
        for row in rows
        if (row.document_id, row.parent_key) in wanted
    }


def _doc_names(db: Session, document_ids: set[str]) -> dict[str, str]:
    if not document_ids:
        return {}
    rows = db.execute(
        select(Document.id, Document.filename).where(Document.id.in_(document_ids))
    ).all()
    return {row[0]: row[1] for row in rows}


def search(
    db: Session,
    query: str,
    *,
    user_id: str,
    knowledge_base_id: str,
    document_ids: list[str] | None = None,
    k: int | None = None,
    merge_parents: bool | None = None,
    debug: bool = False,
    use_reranker: bool | None = None,
) -> dict:
    """执行一次完整检索并返回可直接用于生成/引用的结果。"""
    started = time.time()
    k = k or settings.retrieval_default_k
    merge_parents = settings.merge_parents if merge_parents is None else merge_parents
    depth = max(settings.retrieval_candidate_depth, k * 3)

    dense = qdrant.search_dense(
        query_vector=embed_query(query),
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        document_ids=document_ids,
        limit=depth,
    )
    dense_keys = [(item["document_id"], item["chunk_index"]) for item in dense]
    dense_score = {(item["document_id"], item["chunk_index"]): item["score"] for item in dense}

    index = get_bm25_index(
        db, user_id=user_id, knowledge_base_id=knowledge_base_id, document_ids=document_ids
    )
    bm25_keys, bm25_score = bm25_candidates(index, query, depth)

    fused = rrf_fuse(bm25_keys, dense_keys)
    candidate_keys = fused["order"][:depth]
    chunk_rows = _load_chunks(db, candidate_keys)

    candidates = []
    for key in candidate_keys:
        row = chunk_rows.get(key)
        if row is None:
            continue
        candidates.append(
            {
                **row,
                "rrf_score": round(fused["score"].get(key, 0.0), 6),
                "dense_score": round(dense_score.get(key, 0.0), 4),
                "bm25_score": round(bm25_score.get(key, 0.0), 2),
            }
        )

    if use_reranker is None:
        use_reranker = settings.reranker_enabled
    if use_reranker:
        candidates = reranker.rerank(query, candidates, text_key="text")

    doc_names = _doc_names(db, {candidate["document_id"] for candidate in candidates})

    if not merge_parents:
        results = [
            _child_result(rank, candidate, doc_names)
            for rank, candidate in enumerate(candidates[:k], 1)
        ]
    else:
        results = _parent_results(db, candidates, k, doc_names, debug)

    return {
        "query": query,
        "document_ids": document_ids,
        "knowledge_base_id": knowledge_base_id,
        "merge_parents": merge_parents,
        "elapsed_ms": int((time.time() - started) * 1000),
        "debug": debug,
        "results": results,
    }


def _child_result(rank: int, candidate: dict, doc_names: dict[str, str]) -> dict:
    return {
        "rank": rank,
        "document_id": candidate["document_id"],
        "doc_name": doc_names.get(candidate["document_id"], candidate["document_id"]),
        "chunk_index": candidate["chunk_index"],
        "parent_key": candidate["parent_key"],
        "kind": candidate["kind"],
        "section": candidate["section"],
        "pages": candidate["pages"],
        "page_number": candidate["page_number"],
        "breadcrumb": candidate["breadcrumb"],
        "text": candidate["text"],
        "score": candidate["rrf_score"],
        "dense_score": candidate["dense_score"],
        "bm25_score": candidate["bm25_score"],
        "rerank_score": candidate.get("rerank_score"),
    }


def _parent_results(
    db: Session,
    candidates: list[dict],
    k: int,
    doc_names: dict[str, str],
    debug: bool,
) -> list[dict]:
    groups: dict[tuple[str, str], dict] = {}
    for index, candidate in enumerate(candidates):
        key = (candidate["document_id"], candidate["parent_key"])
        group = groups.setdefault(key, {"first": index, "items": []})
        group["items"].append((index, candidate))

    parent_keys = set(groups)
    parents = _load_parents(db, parent_keys)

    results = []
    budget_used = 0
    for rank, ((document_id, _parent_key), group) in enumerate(
        sorted(groups.items(), key=lambda item: item[1]["first"])[:k], 1
    ):
        best = group["items"][0][1]
        parent = parents.get((document_id, _parent_key))
        parent_text = parent["text"] if parent else None
        text = parent_text or best["text"]

        # 上下文预算：父块超预算时尾部条目降级回 child 原文，不截断、不丢弃
        if parent_text and budget_used + len(parent_text) > settings.context_budget_chars:
            text = best["text"]
        budget_used += len(text)

        result = {
            "rank": rank,
            "document_id": document_id,
            "doc_name": doc_names.get(document_id, document_id),
            "chunk_index": best["chunk_index"],
            "parent_key": _parent_key,
            "kind": best["kind"],
            "section": (parent or best)["section"],
            "pages": (parent or best)["pages"],
            "page_number": (parent or best)["page_number"],
            "breadcrumb": (parent or best)["breadcrumb"],
            "text": text,
            "matched_children": len(group["items"]),
            "score": best["rrf_score"],
            "dense_score": best["dense_score"],
            "bm25_score": best["bm25_score"],
            "rerank_score": best.get("rerank_score"),
        }
        if debug:
            result["children"] = [
                {
                    "chunk_index": item["chunk_index"],
                    "kind": item["kind"],
                    "pages": item["pages"],
                    "rank": index + 1,
                    "score": item["rrf_score"],
                    "dense_score": item["dense_score"],
                    "bm25_score": item["bm25_score"],
                    "rerank_score": item.get("rerank_score"),
                    "text": item["text"][:200],
                }
                for index, item in sorted(group["items"], key=lambda item: item[0])
            ]
        results.append(result)

    return results
