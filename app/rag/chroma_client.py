"""
app/rag/chroma_client.py
"""
from __future__ import annotations

import chromadb

from app.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_client: chromadb.AsyncHttpClient | None = None
_collection: chromadb.AsyncCollection | None = None


async def init_chroma() -> None:
    global _client, _collection

    logger.info(f"Connecting to ChromaDB at {settings.chroma_host}:{settings.chroma_port} ...")

    try:
        _client = await chromadb.AsyncHttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

        await _client.heartbeat()

        _collection = await _client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        count = await _collection.count()
        logger.success(
            f"ChromaDB ready | collection={settings.chroma_collection_name} "
            f"| documents={count}"
        )

        if count == 0:
            logger.warning(
                "ChromaDB collection is empty. "
                "Run: python -m app.rag.ingest to load the knowledge base."
            )

    except Exception as exc:
        logger.warning(f"ChromaDB init failed (non-fatal in dev): {exc}")


def get_collection():
    if _collection is None:
        raise RuntimeError("ChromaDB collection not initialised.")
    return _collection


async def get_collection_stats() -> dict:
    if _collection is None:
        return {"status": "not_initialised", "count": 0}
    count = await _collection.count()
    return {
        "collection_name": settings.chroma_collection_name,
        "document_count": count,
        "status": "ready" if count > 0 else "empty",
    }