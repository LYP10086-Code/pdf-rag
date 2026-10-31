"""文档与检索接口。

- POST /api/documents            上传 PDF（异步入库，202 + job_id）
- GET  /api/documents            列表
- GET  /api/documents/{id}       详情
- DELETE /api/documents/{id}     级联删除（Qdrant → PG → 文件）
- POST /api/documents/{id}/reingest
- GET  /api/jobs/{job_id}        入库进度
- POST /api/search               全库/多文档检索
- POST /api/documents/{id}/search 定向检索
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.retrieval.retriever import search
from app.services import document_service
from app.services.document_service import (
    DocumentNotFound,
    UploadTooLarge,
)

router = APIRouter(prefix="/api", tags=["documents"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int | None = Field(None, ge=1, le=50)
    document_ids: list[str] | None = None
    merge_parents: bool | None = Field(
        None, description="true=父块归并（默认）；false=直接返回检索命中的子块"
    )


@router.post("/documents", status_code=202)
def upload_document(
    file: UploadFile = File(...),  # noqa: B008  FastAPI 依赖注入惯例
    user_id: str = Form(settings.default_user_id),
    knowledge_base_id: str = Form(settings.default_kb_id),
    db: Session = Depends(get_db),
):
    try:
        return document_service.create_document(
            db,
            upload=file,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents")
def list_documents(
    status: str | None = None,
    q: str | None = None,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return document_service.list_documents(db, status=status, query=q, page=page, size=size)


@router.get("/documents/{document_id}")
def get_document(document_id: str, db: Session = Depends(get_db)):
    try:
        return document_service.get_document(db, document_id)
    except DocumentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    try:
        document_service.delete_document(db, document_id)
    except DocumentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": document_id}


@router.post("/documents/{document_id}/reingest", status_code=202)
def reingest_document(document_id: str, db: Session = Depends(get_db)):
    try:
        return document_service.enqueue_reingest(db, document_id)
    except DocumentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    try:
        return document_service.get_job(db, job_id)
    except DocumentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/documents/{document_id}/search")
def search_in_document(
    document_id: str,
    body: SearchRequest,
    debug: bool = Query(False),
    db: Session = Depends(get_db),
):
    try:
        document_service.get_document(db, document_id)
    except DocumentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return search(
        db,
        body.query,
        user_id=settings.default_user_id,
        knowledge_base_id=settings.default_kb_id,
        document_ids=[document_id],
        k=body.k,
        merge_parents=body.merge_parents,
        debug=debug,
    )


@router.post("/search")
def search_documents(
    body: SearchRequest,
    debug: bool = Query(False),
    db: Session = Depends(get_db),
):
    return search(
        db,
        body.query,
        user_id=settings.default_user_id,
        knowledge_base_id=settings.default_kb_id,
        document_ids=body.document_ids,
        k=body.k,
        merge_parents=body.merge_parents,
        debug=debug,
    )
