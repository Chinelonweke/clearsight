"""
app/db/redis_client.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Async Redis client singleton for ClearSight.

Used for:
  - Chat history per session (list, TTL-based)
  - Session metadata (JSONB blob, TTL-based)
  - Rate-limit counters (used by slowapi)
  - Short-lived cache for availability slots

init_redis()   : Called at startup â€” connects and PINGs Redis.
get_redis()    : FastAPI dependency that returns the live client.
close_redis()  : Called at shutdown.
"""

from __future__ import annotations
from typing import AsyncGenerator, Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# â”€â”€ Module-level singleton â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_redis_client: Redis | None = None


async def init_redis() -> None:
    global _redis_client

    logger.info(f"Connecting to Redis at {settings.redis_url} ...")

    try:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            password=settings.redis_password or None,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
        await _redis_client.ping()
        logger.success("Redis connection verified.")

    except Exception as exc:
        logger.warning(
            f"Redis unavailable (non-fatal): {exc}. "
            f"Chat history disabled until Redis is reachable."
        )
        _redis_client = None


async def get_redis() -> AsyncGenerator[Redis, None]:
    if _redis_client is None:
        raise RuntimeError("Redis is not available.")
    yield _redis_client


async def close_redis() -> None:
    """Gracefully close the Redis connection pool on shutdown."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis connection closed.")


def get_redis_client() -> Redis:
    if _redis_client is None:
        raise RuntimeError("Redis is not available.")
    return _redis_client

