"""
app/models/database.py
─────────────────────────────────────────────────────────
Async SQLAlchemy engine connected to NeonDB (PostgreSQL).

Exports:
  - Base        : declarative base all ORM models inherit from
  - engine      : async engine singleton
  - AsyncSessionLocal : async session factory
  - get_db()    : FastAPI dependency that yields a session per request
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """
    Declarative base for all ORM models.
    All models in app/models/*.py inherit from this.
    """
    pass


# ── Engine ────────────────────────────────────────────────────────────────────
# pool_size / max_overflow are tuned for a small clinic:
#   - 5 persistent connections in the pool
#   - 10 extra connections allowed under spike load
#   - connections recycled every 30 minutes (prevents NeonDB idle timeouts)
engine = create_async_engine(
    settings.database_url,
    echo=settings.is_development,       # log SQL in dev; silent in prod
    pool_size=5,
    max_overflow=10,
    pool_recycle=1800,                  # 30-minute recycle to avoid stale connections
    pool_pre_ping=True,                 # test connection liveness before use
)

# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,             # keep ORM objects usable after commit
    autoflush=False,
    autocommit=False,
)


# ── FastAPI dependency ─────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session for a single request.
    Automatically commits on success and rolls back on any exception.

    Use as a FastAPI dependency:
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise