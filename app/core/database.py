"""SQLAlchemy 引擎与会话（惰性初始化）。

PostgreSQL 保存文档、父块、子块和任务状态；向量与文件都是可重建的派生产物。
引擎在首次访问时才创建，便于只做纯函数测试时不要求数据库驱动/服务在线。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


@lru_cache
def get_engine() -> Engine:
    return create_engine(
        settings.database_url,
        echo=settings.db_echo,
        pool_pre_ping=True,
    )


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        expire_on_commit=False,
    )


def get_db() -> Iterator[Session]:
    """FastAPI 依赖：每个请求一个会话。"""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """脚本/后台任务用的事务上下文：成功提交，异常回滚。"""
    db = get_session_factory()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """建表。表结构以 models 为准，启动时幂等执行。"""
    from app.core import models  # noqa: F401  确保模型已注册到 Base.metadata

    Base.metadata.create_all(get_engine())
