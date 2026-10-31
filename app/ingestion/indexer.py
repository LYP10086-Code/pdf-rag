"""入库编排：解析 → 分块 → 向量化 → 落库。

状态机：pending → parsing → chunking → indexing → ready
任一步失败 → failed（错误写入 documents.error 与 ingest_jobs.error，可重试）。

一致性约定：文档记录以 PostgreSQL 为准；重入库先删该文档的 Qdrant 点与旧分块，
再重建，避免半成品数据。
"""

import json
import logging
import traceback
from datetime import datetime, timezone

from sqlalchemy import delete

from app.core import qdrant
from app.core.config import settings
from app.core.database import session_scope
from app.core.embeddings import embed_texts
from app.core.models import (
    Chunk,
    Document,
    DocumentStatus,
    IngestJob,
    JobStatus,
    ParentChunk,
)
from app.ingestion.chunker import build_chunks
from app.ingestion.parser import parse_pdf
from app.retrieval.hybrid import invalidate_bm25_cache

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _set_stage(document_id: str, job_id: str, status: str, step: str, progress: int) -> None:
    with session_scope() as db:
        job = db.get(IngestJob, job_id)
        document = db.get(Document, document_id)
        if job:
            job.status = JobStatus.RUNNING
            job.step = step
            job.progress = progress
        if document:
            document.status = status
            document.updated_at = _now()


def _write_artifacts(
    document_id: str, markdown: str, chunks: list[dict], parents: list[dict]
) -> None:
    """落盘解析/分块产物，便于人工核对与回放。"""
    base = settings.markdown_dir
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{document_id}.md").write_text(markdown, encoding="utf-8")
    for name, rows in (("chunks", chunks), ("parents", parents)):
        path = base / f"{document_id}.{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_ingest_job(job_id: str, document_id: str) -> None:
    try:
        _set_stage(document_id, job_id, DocumentStatus.PARSING, "parsing", 10)
        with session_scope() as db:
            document = db.get(Document, document_id)
            if document is None:
                raise ValueError(f"文档不存在：{document_id}")
            parsed = parse_pdf(
                document.storage_path,
                work_dir=settings.markdown_dir / document_id,
                images_dir=settings.images_dir / document_id,
            )
            document.page_count = parsed.page_count
            document.updated_at = _now()

        _write_artifacts(document_id, parsed.markdown, [], [])
        logger.info("文档 %s 解析完成：%s", document_id, parsed.stats)
        _set_stage(document_id, job_id, DocumentStatus.CHUNKING, "chunking", 50)
        with session_scope() as db:
            document = db.get(Document, document_id)
            if document is None:
                raise ValueError(f"文档不存在：{document_id}")
            result = build_chunks(parsed, document_id, document_title=document.filename)

        _write_artifacts(document_id, parsed.markdown, result.chunks, result.parents)
        logger.info("文档 %s 分块完成：%s", document_id, result.stats)

        _set_stage(document_id, job_id, DocumentStatus.INDEXING, "indexing", 75)
        with session_scope() as db:
            document = db.get(Document, document_id)
            if document is None:
                raise ValueError(f"文档不存在：{document_id}")
            filename, user_id, knowledge_base_id = (
                document.filename,
                document.user_id,
                document.knowledge_base_id,
            )

        qdrant.delete_document(document_id)
        vectors = embed_texts([chunk["embed_text"] for chunk in result.chunks])
        qdrant.upsert_chunks(
            document_id=document_id,
            filename=filename,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            chunks=result.chunks,
            vectors=vectors,
        )

        with session_scope() as db:
            document = db.get(Document, document_id)
            if document is None:
                raise ValueError(f"文档不存在：{document_id}")

            db.execute(delete(Chunk).where(Chunk.document_id == document_id))
            db.execute(delete(ParentChunk).where(ParentChunk.document_id == document_id))

            db.add_all(
                [
                    ParentChunk(
                        document_id=document_id,
                        parent_key=parent["parent_key"],
                        section=parent["section"],
                        pages=parent["pages"],
                        page_number=parent["page_number"],
                        breadcrumb=parent["breadcrumb"],
                        text=parent["text"],
                    )
                    for parent in result.parents
                ]
            )
            db.add_all(
                [
                    Chunk(
                        document_id=document_id,
                        chunk_index=chunk["chunk_index"],
                        parent_key=chunk["parent_key"],
                        kind=chunk["kind"],
                        section=chunk["section"],
                        pages=chunk["pages"],
                        page_number=chunk["page_number"],
                        breadcrumb=chunk["breadcrumb"],
                        text=chunk["text"],
                        embed_text=chunk["embed_text"],
                        content_hash=chunk["content_hash"],
                        qdrant_point_id=qdrant.point_id(document_id, chunk["chunk_index"]),
                    )
                    for chunk in result.chunks
                ]
            )
            document.status = DocumentStatus.READY
            document.chunk_count = len(result.chunks)
            document.parent_count = len(result.parents)
            document.index_version = settings.embedding_model
            document.error = None
            document.updated_at = _now()

            job = db.get(IngestJob, job_id)
            if job:
                job.status = JobStatus.DONE
                job.step = "done"
                job.progress = 100
                job.ended_at = _now()
        invalidate_bm25_cache()
        logger.info(
            "文档 %s 入库完成：%d 子块 / %d 父块",
            document_id,
            len(result.chunks),
            len(result.parents),
        )
    except Exception as exc:  # noqa: BLE001  后台任务必须兜底并写状态
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("文档 %s 入库失败", document_id)
        with session_scope() as db:
            document = db.get(Document, document_id)
            if document:
                document.status = DocumentStatus.FAILED
                document.error = error
                document.updated_at = _now()
            job = db.get(IngestJob, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.step = "failed"
                job.progress = 0
                job.ended_at = _now()
                job.error = error + "\n" + traceback.format_exc(limit=3)
