"""
app/services/rag_service.py
--------------------------------------------------
RAG retrieval service.
1. Embed the patient query with fastembed (all-MiniLM-L6-v2, ONNX).
2. Query ChromaDB for the top-k most relevant eye condition chunks.
3. Return the chunks as context for the LLM.

fastembed produces identical 384-dim vectors to sentence-transformers
for the same model - no re-ingestion needed if vectors were already
created with all-MiniLM-L6-v2.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

# Module-level model cache - loaded once, reused across requests
_embed_model = None


def _get_embed_model():
    """Lazy-load fastembed model (ONNX, no PyTorch required)."""
    global _embed_model
    if _embed_model is None:
        try:
            from fastembed import TextEmbedding  # type: ignore
            logger.info("Loading fastembed model (all-MiniLM-L6-v2)...")
            _embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            logger.success("fastembed model loaded (384-dim ONNX).")
        except Exception as exc:
            logger.error(f"Failed to load fastembed model: {exc}")
            raise
    return _embed_model


def _embed_query(text: str) -> list[float]:
    """
    Embed a query string into a 384-dimensional vector using fastembed.
    fastembed.embed() returns a generator of numpy arrays.
    """
    model = _get_embed_model()
    embeddings = list(model.embed([text]))  # returns list of np.ndarray
    return embeddings[0].tolist()


class RAGService:
    """
    Retrieval-Augmented Generation service.
    Embeds patient queries and retrieves relevant eye condition context
    from ChromaDB to ground the LLM responses.
    """

    def __init__(self) -> None:
        from app.rag.chroma_client import get_collection as get_chroma_collection
        self._collection = get_chroma_collection()
        logger.info("RAGService initialised.")

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[str]:
        """
        Retrieve the top-k most relevant chunks for a given query.

        Args:
            query:  Patient message or symptom description.
            top_k:  Number of chunks to return.

        Returns:
            List of text chunks to use as LLM context.
        """
        if not query or not query.strip():
            return []

        try:
            # Embed in thread executor - fastembed is synchronous
            loop = asyncio.get_running_loop()
            query_embedding = await loop.run_in_executor(
                None, _embed_query, query.strip()
            )

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count() or 1),
                include=["documents"],
            )

            docs = results.get("documents", [[]])[0]
            logger.debug(
                f"RAG retrieved {len(docs)} chunks | "
                f"query={query[:50]!r}"
            )
            return docs

        except Exception as exc:
            logger.warning(f"RAG retrieval failed (non-fatal): {exc}")
            return []