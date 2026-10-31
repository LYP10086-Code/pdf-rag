"""问答接口：RAG 链与工具调用 Agent。"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.services.chat_service import answer_question

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[str] | None = None
    k: int | None = Field(None, ge=1, le=20)
    mode: Literal["rag", "agent"] | None = None
    merge_parents: bool | None = None
    debug: bool = False


@router.post("/chat")
def chat(body: ChatRequest, db: Session = Depends(get_db)):
    try:
        return answer_question(
            db,
            body.question,
            user_id=settings.default_user_id,
            knowledge_base_id=settings.default_kb_id,
            document_ids=body.document_ids,
            k=body.k,
            mode=body.mode,
            merge_parents=body.merge_parents,
            debug=body.debug,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
