"""问答服务：检索 → RAG 链或工具调用 Agent → 答案 + 服务端引用。"""

import logging
import time

from sqlalchemy.orm import Session

from app.core.config import settings
from app.rag.agent import ask_agent
from app.rag.chain import (
    NONE_ANSWER,
    answer_with_context,
    build_citations,
    format_context,
)
from app.retrieval.retriever import search

logger = logging.getLogger(__name__)

VALID_MODES = {"rag", "agent"}


def answer_question(
    db: Session,
    question: str,
    *,
    user_id: str,
    knowledge_base_id: str,
    document_ids: list[str] | None = None,
    k: int | None = None,
    mode: str | None = None,
    merge_parents: bool | None = None,
    debug: bool = False,
) -> dict:
    selected_mode = (mode or settings.chat_mode).lower()
    if selected_mode not in VALID_MODES:
        raise ValueError("mode 仅支持 rag|agent")

    started = time.time()
    retrieval_meta = {"document_ids": document_ids, "k": k}
    results: list[dict] = []

    if selected_mode == "agent":
        output = ask_agent(
            question,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            document_ids=document_ids,
            k=k,
        )
        results = output["results"]
        answer = output["answer"]
        retrieval_meta["agent_steps"] = output["steps"]
    else:
        payload = search(
            db,
            question,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            document_ids=document_ids,
            k=k,
            merge_parents=merge_parents,
            debug=debug,
        )
        results = payload["results"]
        retrieval_meta["elapsed_ms"] = payload["elapsed_ms"]
        if not results:
            answer = NONE_ANSWER
        else:
            answer = answer_with_context(question, format_context(results))

    return {
        "question": question,
        "mode": selected_mode,
        "answer": answer,
        "citations": build_citations(results),
        "retrieval": retrieval_meta,
        "elapsed_ms": int((time.time() - started) * 1000),
    }
