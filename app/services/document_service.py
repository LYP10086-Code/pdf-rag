"""文档服务：上传、命名冲突、入库任务、列表、删除。

不变量：
- 物理路径不使用用户文件名，统一 `storage/originals/{doc_id}/source.pdf`
- 上传只落原件 + 建任务，解析/向量化全部异步
- 文档状态以 PostgreSQL 为准；Qdrant 与文件是可重建的派生产物
"""

import hashlib
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile
from pypdf import PdfReader
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core import qdrant
from app.core.config import settings
from app.core.models import (
    Chunk,
    Document,
    DocumentStatus,
    IngestJob,
    JobStatus,
    ParentChunk,
    new_id,
)
from app.retrieval.hybrid import invalidate_bm25_cache

logger = logging.getLogger(__name__)

_CTRL_RE = re.compile(r"[\x00-\x1f]")
_SAFE_RE = re.compile(r'[\\/:*?"<>|]')


class UploadTooLarge(ValueError):
    pass


class DocumentNotFound(LookupError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def sanitize_name(original: str) -> str:
    name = Path(original.replace("\\", "/")).name
    name = _CTRL_RE.sub("", name)
    name = _SAFE_RE.sub("_", name).strip()
    return name or "document.pdf"


def _save_upload(upload: UploadFile, target: Path) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    size = 0
    with target.open("wb") as handle:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_bytes:
                handle.close()
                target.unlink(missing_ok=True)
                raise UploadTooLarge(
                    f"文件超过上限 {settings.max_upload_mb}MB"
                )
            sha.update(chunk)
            handle.write(chunk)
    return sha.hexdigest(), size


def _resolve_filename(db: Session, original: str) -> str:
    """同名文档加时间后缀，避免 UNIQUE(filename) 冲突。"""
    if not db.execute(
        select(Document.id).where(Document.filename == original)
    ).first():
        return original

    base, ext = original.rsplit(".", 1) if "." in original else (original, "")
    ext = f".{ext}" if ext else ""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = f"{base}_{stamp}{ext}"
    index = 2
    while db.execute(select(Document.id).where(Document.filename == candidate)).first():
        candidate = f"{base}_{stamp}_{index}{ext}"
        index += 1
    return candidate


def create_document(
    db: Session,
    *,
    upload: UploadFile,
    user_id: str,
    knowledge_base_id: str,
) -> dict:
    original = sanitize_name(upload.filename or "document.pdf")
    if not original.lower().endswith(".pdf"):
        raise ValueError("仅支持 PDF 文件")

    filename = _resolve_filename(db, original)
    document_id = new_id("d")
    document = Document(
        id=document_id,
        filename=filename,
        original_name=original,
        storage_path="",
        sha256="",
        size_bytes=0,
        status=DocumentStatus.PENDING,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
    )
    path = settings.originals_dir / document_id / "source.pdf"

    try:
        sha256, size = _save_upload(upload, path)
        page_count = len(PdfReader(str(path)).pages)
    except Exception:
        shutil.rmtree(path.parent, ignore_errors=True)
        raise

    document.storage_path = str(path)
    document.sha256 = sha256
    document.size_bytes = size
    document.page_count = page_count
    db.add(document)
    db.flush()

    job = _create_job(db, document.id)
    db.commit()

    # 解析交给独立 worker 容器从队列认领执行，api 不再进程内跑任务
    logger.info("文档 %s 上传完成，任务 %s 入队", document.id, job.id)
    return {
        "id": document.id,
        "filename": document.filename,
        "status": document.status,
        "page_count": page_count,
        "size_bytes": size,
        "job_id": job.id,
    }


def _create_job(db: Session, document_id: str) -> IngestJob:
    job = IngestJob(
        document_id=document_id,
        status=JobStatus.QUEUED,
        step="queued",
        progress=0,
        started_at=_now(),
    )
    db.add(job)
    db.flush()
    return job


def enqueue_reingest(db: Session, document_id: str) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise DocumentNotFound("文档不存在")
    job = _create_job(db, document_id)
    document.status = DocumentStatus.PENDING
    document.error = None
    document.updated_at = _now()
    db.commit()
    # 重新入队，由 worker 认领执行
    return {"document_id": document_id, "job_id": job.id, "status": document.status}


def list_documents(
    db: Session,
    *,
    status: str | None = None,
    query: str | None = None,
    page: int = 1,
    size: int = 20,
) -> dict:
    stmt = select(Document)
    if status:
        stmt = stmt.where(Document.status == status)
    if query:
        stmt = stmt.where(Document.filename.ilike(f"%{query}%"))
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = (
        db.execute(
            stmt.order_by(Document.created_at.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
        .scalars()
        .all()
    )
    return {
        "items": [_document_dict(document, include_path=False) for document in rows],
        "total": total,
        "page": page,
        "size": size,
    }


def get_document(db: Session, document_id: str) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise DocumentNotFound("文档不存在")
    return _document_dict(document, include_path=True)


def get_job(db: Session, job_id: str) -> dict:
    job = db.get(IngestJob, job_id)
    if job is None:
        raise DocumentNotFound("任务不存在")
    document = db.get(Document, job.document_id)
    return {
        "id": job.id,
        "document_id": job.document_id,
        "status": job.status,
        "step": job.step,
        "progress": job.progress,
        "error": job.error,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "document": _document_dict(document, include_path=False) if document else None,
    }


def delete_document(db: Session, document_id: str) -> None:
    document = db.get(Document, document_id)
    if document is None:
        raise DocumentNotFound("文档不存在")

    qdrant.delete_document(document_id)
    db.execute(delete(IngestJob).where(IngestJob.document_id == document_id))
    db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    db.execute(delete(ParentChunk).where(ParentChunk.document_id == document_id))
    db.delete(document)
    db.commit()

    shutil.rmtree(settings.originals_dir / document_id, ignore_errors=True)
    invalidate_bm25_cache()
    logger.info("文档 %s 已删除", document_id)


def recover_stale_jobs(db: Session) -> int:
    """服务重启后把残留的排队/运行中任务标为失败，避免前端一直等待。"""
    jobs = (
        db.execute(
            select(IngestJob).where(
                IngestJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
            )
        )
        .scalars()
        .all()
    )
    for job in jobs:
        job.status = JobStatus.FAILED
        job.step = "failed"
        job.error = "服务重启，任务中断"
        job.ended_at = _now()

    documents = (
        db.execute(
            select(Document).where(
                Document.status.in_(
                    [
                        DocumentStatus.PENDING,
                        DocumentStatus.PARSING,
                        DocumentStatus.CHUNKING,
                        DocumentStatus.INDEXING,
                    ]
                )
            )
        )
        .scalars()
        .all()
    )
    for document in documents:
        document.status = DocumentStatus.FAILED
        document.error = "服务重启，任务中断"
        document.updated_at = _now()
    db.commit()
    return len(jobs)


def shutdown() -> None:
    _executor.shutdown(wait=False, cancel_futures=True)


def _document_dict(document: Document, *, include_path: bool) -> dict:
    data = {
        "id": document.id,
        "filename": document.filename,
        "original_name": document.original_name,
        "file_type": document.file_type,
        "status": document.status,
        "page_count": document.page_count,
        "chunk_count": document.chunk_count,
        "parent_count": document.parent_count,
        "size_bytes": document.size_bytes,
        "size_mb": round(document.size_bytes / (1024 * 1024), 1),
        "sha256": document.sha256,
        "error": document.error,
        "index_version": document.index_version,
        "knowledge_base_id": document.knowledge_base_id,
        "user_id": document.user_id,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }
    if include_path:
        data["storage_path"] = document.storage_path
    return data
