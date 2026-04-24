"""
app/rag/chunking.py
--------------------------------------------------
Splits markdown documents into overlapping chunks and embeds them.

Pipeline:
  1. Split document into sentences.
  2. Group sentences into overlapping chunks (~200 tokens each).
  3. Encode each chunk with fastembed (all-MiniLM-L6-v2, ONNX).
     Produces identical 384-dim vectors to sentence-transformers.
"""

from __future__ import annotations

import re
from typing import Iterator

from app.core.logger import get_logger

logger = get_logger(__name__)

# Chunk configuration
CHUNK_SIZE = 5        # sentences per chunk
CHUNK_OVERLAP = 2     # sentences of overlap between chunks

# Module-level model cache
_embed_model = None


def _get_embed_model():
    """Lazy-load fastembed model to avoid slow import at module level."""
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding  # type: ignore
        logger.info("Loading fastembed model (all-MiniLM-L6-v2)...")
        _embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.success("fastembed model loaded.")
    return _embed_model


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _make_chunks(sentences: list[str]) -> list[str]:
    """Group sentences into overlapping chunks."""
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i: i + CHUNK_SIZE]
        chunks.append(" ".join(chunk))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def chunk_document(text: str, doc_id: str) -> list[dict]:
    """
    Split a document into chunks and embed each one.

    Returns:
        List of dicts with keys: id, text, embedding
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = _make_chunks(sentences)
    model = _get_embed_model()

    # fastembed.embed() accepts a list and returns a generator
    embeddings = list(model.embed(chunks))

    results = []
    for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        results.append({
            "id": f"{doc_id}_chunk_{i}",
            "text": chunk_text,
            "embedding": embedding.tolist(),
        })

    logger.debug(f"Chunked {doc_id} into {len(results)} chunks.")
    return results