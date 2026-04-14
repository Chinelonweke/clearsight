"""
app/rag/ingest.py
─────────────────────────────────────────────────────────
One-shot ingestion script: loads all markdown files from
data/knowledge_base/ into ChromaDB with semantic chunking.

Run once at setup, then again whenever you add a new .md file:
    python -m app.rag.ingest

The script is idempotent: it deletes and recreates all documents
for a given source file, so re-running after editing a .md file
is safe and will not create duplicates.

Output example:
    [INFO] Loading glaucoma.md → 18 chunks
    [INFO] Loading cataracts.md → 16 chunks
    ...
    [SUCCESS] Ingestion complete | 14 files | 212 total chunks
"""

import asyncio
import hashlib
import re
import sys
from pathlib import Path

# Ensure the project root is on the path when run as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import chromadb
from sentence_transformers import SentenceTransformer  # type: ignore

from app.config import settings
from app.core.logger import get_logger
from app.rag.chunking import chunk_document_with_metadata

logger = get_logger(__name__)

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _condition_name_from_filename(filename: str) -> str:
    """Convert 'diabetic_retinopathy.md' → 'Diabetic Retinopathy'"""
    stem = Path(filename).stem
    return stem.replace("_", " ").title()


def _make_chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic, stable ID for each chunk — safe to re-ingest."""
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


async def ingest_knowledge_base(reset: bool = False) -> dict:
    """
    Main ingestion function.

    Args:
        reset: If True, wipe the entire collection before ingesting.
               Use this for a clean rebuild.

    Returns:
        Summary dict with file_count and total_chunks.
    """
    logger.info("=" * 60)
    logger.info("  ClearSight — Knowledge Base Ingestion")
    logger.info("=" * 60)

    # ── Load embedding model ───────────────────────────────────────────────────
    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    logger.success("Embedding model loaded.")

    # ── Connect to ChromaDB ────────────────────────────────────────────────────
    logger.info(f"Connecting to ChromaDB at {settings.chroma_host}:{settings.chroma_port}")
    client = await chromadb.AsyncHttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    await client.heartbeat()

    if reset:
        logger.warning(f"Resetting collection '{settings.chroma_collection_name}'...")
        try:
            await client.delete_collection(settings.chroma_collection_name)
        except Exception:
            pass

    collection = await client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.success(f"Collection '{settings.chroma_collection_name}' ready.")

    # ── Discover markdown files ────────────────────────────────────────────────
    md_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))
    if not md_files:
        logger.error(f"No .md files found in {KNOWLEDGE_BASE_DIR}")
        return {"file_count": 0, "total_chunks": 0}

    logger.info(f"Found {len(md_files)} knowledge base file(s):")
    for f in md_files:
        logger.info(f"  • {f.name}")

    total_chunks = 0

    # ── Ingest each file ───────────────────────────────────────────────────────
    for md_file in md_files:
        source = md_file.name
        condition = _condition_name_from_filename(source)

        logger.info(f"\nProcessing: {source} ({condition})")
        text = md_file.read_text(encoding="utf-8")

        # Semantic chunking with metadata
        chunk_dicts = chunk_document_with_metadata(
            text=text,
            source_file=source,
            condition_name=condition,
        )

        if not chunk_dicts:
            logger.warning(f"  No chunks produced for {source} — skipping.")
            continue

        # Delete existing docs for this source (idempotent re-ingest)
        try:
            existing = await collection.get(where={"source": source})
            if existing["ids"]:
                await collection.delete(ids=existing["ids"])
                logger.debug(f"  Deleted {len(existing['ids'])} existing chunks for {source}")
        except Exception as exc:
            logger.debug(f"  No existing chunks to delete for {source}: {exc}")

        # Embed all chunks for this file in one batch (faster than one-by-one)
        texts = [c["text"] for c in chunk_dicts]
        embeddings = embed_model.encode(texts, show_progress_bar=False).tolist()

        ids = [_make_chunk_id(source, c["chunk_index"]) for c in chunk_dicts]
        metadatas = [
            {
                "source": c["source"],
                "condition": c["condition"],
                "section": c["section"],
                "chunk_index": c["chunk_index"],
                "char_count": c["char_count"],
            }
            for c in chunk_dicts
        ]

        # Add to ChromaDB in batches of 50
        batch_size = 50
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            await collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

        logger.success(f"  ✓ {source} → {len(chunk_dicts)} chunks ingested")
        total_chunks += len(chunk_dicts)

    # ── Final report ───────────────────────────────────────────────────────────
    final_count = await collection.count()
    logger.info("=" * 60)
    logger.success(
        f"Ingestion complete | "
        f"files={len(md_files)} | chunks_added={total_chunks} | "
        f"collection_total={final_count}"
    )
    logger.info("=" * 60)

    return {"file_count": len(md_files), "total_chunks": total_chunks}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest ClearSight knowledge base into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Wipe collection before ingesting")
    args = parser.parse_args()

    asyncio.run(ingest_knowledge_base(reset=args.reset))