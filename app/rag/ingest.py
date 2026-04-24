"""
app/rag/ingest.py
--------------------------------------------------
Ingests eye condition markdown documents into ChromaDB.

Run once (or after adding new documents):
    docker exec clearsight_api python3 -m app.rag.ingest

After switching from sentence-transformers to fastembed:
    1. Clear the existing ChromaDB collection (same vectors, safe to re-ingest)
    2. Run this script to re-ingest all documents
    Both libraries use all-MiniLM-L6-v2 producing identical 384-dim vectors.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.core.logger import get_logger
from app.rag.chunking import chunk_document
from app.rag.chroma_client import get_chroma_collection

logger = get_logger(__name__)

# Path to eye condition markdown files
DOCS_DIR = Path(__file__).parent.parent.parent / "data" / "knowledge_base"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def ingest_all(clear_first: bool = True) -> int:
    """
    Ingest all markdown documents from DOCS_DIR into ChromaDB.

    Args:
        clear_first: If True, wipe existing collection before ingesting.
                     Recommended when switching embedding libraries.

    Returns:
        Total number of chunks ingested.
    """
    collection = get_chroma_collection()

    if clear_first:
        existing = collection.count()
        if existing > 0:
            logger.info(f"Clearing {existing} existing chunks from ChromaDB...")
            # Get all IDs and delete them
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
            logger.success("ChromaDB collection cleared.")

    if not DOCS_DIR.exists():
        logger.error(f"Documents directory not found: {DOCS_DIR}")
        return 0

    md_files = list(DOCS_DIR.glob("*.md"))
    if not md_files:
        logger.warning(f"No markdown files found in {DOCS_DIR}")
        return 0

    logger.info(f"Ingesting {len(md_files)} documents from {DOCS_DIR}...")

    total_chunks = 0

    for md_file in sorted(md_files):
        try:
            text = md_file.read_text(encoding="utf-8")
            doc_id = md_file.stem

            chunks = chunk_document(text, doc_id)
            if not chunks:
                logger.warning(f"No chunks produced for {md_file.name}")
                continue

            collection.add(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                embeddings=[c["embedding"] for c in chunks],
                metadatas=[{"source": md_file.name, "doc_id": doc_id} for c in chunks],
            )

            total_chunks += len(chunks)
            logger.success(f"Ingested {md_file.name} -> {len(chunks)} chunks")

        except Exception as exc:
            logger.error(f"Failed to ingest {md_file.name}: {exc}")
            continue

    logger.success(
        f"Ingestion complete | documents={len(md_files)} | chunks={total_chunks}"
    )
    return total_chunks


if __name__ == "__main__":
    import asyncio
    from app.db.neon import init_db

    async def main():
        await init_db()
        total = ingest_all(clear_first=True)
        print(f"\nIngested {total} chunks total.")

    asyncio.run(main())
