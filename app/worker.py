"""入库任务 Worker：从 Postgres 队列认领 ingest_jobs 并执行解析流水线。

与 api 共用镜像（仅启动命令不同），在独立容器运行（compose 的 worker 服务），
解析重活不再占用 api 进程的 CPU/内存配额。设计要点：

- 认领：SELECT ... FOR UPDATE SKIP LOCKED 原子认领，天然支持多 worker 副本
- 租约：claimed_at 由主循环心跳续期；超时未续期（worker 崩溃/被杀）的任务
  自动重置回 queued，由后续循环重新认领
- 隔离：api 容器重建/重启不影响正在执行的入库任务
- 探活：循环内 touch 心跳文件，compose healthcheck 检查文件新鲜度

env（直接读环境变量，不进 Settings）：
    WORKER_POLL_INTERVAL_S   队列轮询间隔（默认 1）
    WORKER_LEASE_SECONDS     租约时长（默认 900，需大于最慢解析耗时）
    WORKER_HEARTBEAT_SECONDS 心跳间隔（默认 30）
"""

import logging
import os
import signal
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select, update

from app.core import models  # noqa: F401  确保模型注册
from app.core.config import settings
from app.core.database import get_engine, init_db, session_scope
from app.core.models import Document, DocumentStatus, IngestJob, JobStatus
from app.ingestion.indexer import run_ingest_job

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = float(os.environ.get("WORKER_POLL_INTERVAL_S", "1"))
LEASE_SECONDS = int(os.environ.get("WORKER_LEASE_SECONDS", "900"))
HEARTBEAT_SECONDS = float(os.environ.get("WORKER_HEARTBEAT_SECONDS", "30"))
HEARTBEAT_FILE = Path(os.environ.get("WORKER_HEARTBEAT_FILE", "/tmp/worker-heartbeat"))

_stop = threading.Event()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _touch_heartbeat() -> None:
    try:
        HEARTBEAT_FILE.touch()
    except OSError:
        logger.warning("心跳文件写入失败：%s", HEARTBEAT_FILE)


def _requeue_stale() -> int:
    """把租约超时的 running 任务重置回 queued（worker 崩溃/被杀的残留）。"""
    with session_scope() as db:
        job_ids = db.execute(
            update(IngestJob)
            .where(
                IngestJob.status == JobStatus.RUNNING,
                IngestJob.claimed_at < _now() - timedelta(seconds=LEASE_SECONDS),
            )
            .values(status=JobStatus.QUEUED, claimed_at=None, step="queued", progress=0)
            .returning(IngestJob.document_id)
        ).scalars().all()
        if job_ids:
            db.execute(
                update(Document)
                .where(Document.id.in_(job_ids))
                .values(status=DocumentStatus.PENDING)
            )
    return len(job_ids)


def _claim_job() -> tuple[str, str] | None:
    """原子认领一个 queued 任务；无任务返回 None。"""
    with session_scope() as db:
        row = db.execute(
            select(IngestJob)
            .where(IngestJob.status == JobStatus.QUEUED)
            .order_by(IngestJob.started_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        ).scalar_one_or_none()
        if row is None:
            return None
        row.status = JobStatus.RUNNING
        row.claimed_at = _now()
        return row.id, row.document_id


def _heartbeat(job_id: str) -> None:
    try:
        with session_scope() as db:
            db.execute(
                update(IngestJob).where(IngestJob.id == job_id).values(claimed_at=_now())
            )
    except Exception:  # noqa: BLE001  心跳失败不影响任务，靠租约兜底
        logger.warning("任务 %s 心跳失败", job_id, exc_info=True)


def _run_with_heartbeat(job_id: str, document_id: str) -> None:
    """在子线程执行流水线，主线程按固定间隔续租。"""
    thread = threading.Thread(
        target=run_ingest_job, args=(job_id, document_id), name=f"ingest-{job_id}"
    )
    thread.start()
    while True:
        thread.join(timeout=HEARTBEAT_SECONDS)
        if not thread.is_alive():
            break
        _heartbeat(job_id)
        _touch_heartbeat()
    thread.join()


def _handle_signal(signum: int, _frame: object) -> None:
    logger.warning("收到信号 %s，当前任务完成后退出", signum)
    _stop.set()


def main() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    init_db()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    logger.info(
        "worker 启动：poll=%ss lease=%ss heartbeat=%ss",
        POLL_INTERVAL_S,
        LEASE_SECONDS,
        HEARTBEAT_SECONDS,
    )

    try:
        while not _stop.is_set():
            _touch_heartbeat()
            requeued = _requeue_stale()
            if requeued:
                logger.warning("重置 %d 个租约超时任务回队列", requeued)
            claimed = _claim_job()
            if claimed is None:
                _stop.wait(POLL_INTERVAL_S)
                continue
            job_id, document_id = claimed
            logger.info("认领任务 %s（文档 %s）", job_id, document_id)
            _run_with_heartbeat(job_id, document_id)
            logger.info("任务 %s 处理结束", job_id)
    finally:
        get_engine().dispose()
        logger.info("worker 退出")


if __name__ == "__main__":
    main()
