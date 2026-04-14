from __future__ import annotations
"""
app/services/rag_service.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RAG (Retrieval-Augmented Generation) service.

At query time:
  1. Embed the patient's query with sentence-transformers.
  2. Retrieve the top-k most semantically similar chunks from ChromaDB.
  3. Return the chunk texts for injection into the LLM system prompt.

The RAG service does NOT call the LLM â€” it only retrieves.
The LLMService.chat_with_context() method handles injection.

Usage:
    rag = RAGService()
    chunks = await rag.retrieve("sudden loss of vision left eye", top_k=3)
    # Returns list of relevant text excerpts from the knowledge base
"""

from typing import List, Optional
import asyncio
from app.core.exceptions import RAGError
from app.core.logger import get_logger
from app.rag.chroma_client import get_collection

logger = get_logger(__name__)

# Lazy-load to avoid slow import at startup
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


class RAGService:
    """
    Async retrieval from ChromaDB for clinical context augmentation.
    """

    def __init__(self) -> None:
        logger.info("RAGService initialised.")

    def _embed_query(self, query: str) -> list[float]:
        """Embed a query string into a vector using sentence-transformers."""
        model = _get_embed_model()
        vector = model.encode([query], show_progress_bar=False)[0]
        return vector.tolist()

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
        condition_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
        min_relevance_threshold: float = 0.3,
    ) -> List[str]:
        """
        Retrieve the most relevant clinical knowledge chunks for a patient query.

        Args:
            query:                  Patient symptoms text or triage question.
            top_k:                  Number of chunks to retrieve.
            condition_filter:       Restrict retrieval to a specific condition
                                    (e.g. "Glaucoma"). Use when condition is known.
            section_filter:         Restrict to a section type (e.g. "Triage guidance").
            min_relevance_threshold: Discard chunks with cosine distance > this.
                                    ChromaDB returns distance (lower = more similar).
                                    0.3 threshold means distance < 0.7.

        Returns:
            List of chunk text strings, ordered by relevance (most relevant first).
            Returns empty list if ChromaDB is not available or no results found.
        """
        if not query.strip():
            return []

        try:
            collection = get_collection()
        except RuntimeError as exc:
            logger.warning(f"RAG unavailable: {exc}")
            return []

        # Build optional metadata filter
        where_filter: dict | None = None
        if condition_filter and section_filter:
            where_filter = {
                "$and": [
                    {"condition": condition_filter},
                    {"section": section_filter},
                ]
            }
        elif condition_filter:
            where_filter = {"condition": condition_filter}
        elif section_filter:
            where_filter = {"section": section_filter}

        try:
            loop = asyncio.get_event_loop()

            # Embed the query in a thread executor (sentence-transformers is sync)
            query_vector = await loop.run_in_executor(None, self._embed_query, query)

            query_kwargs = {
                "query_embeddings": [query_vector],
                "n_results": top_k,
                "include": ["documents", "distances", "metadatas"],
            }
            if where_filter:
                query_kwargs["where"] = where_filter

            results = await collection.query(**query_kwargs)

            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            # Filter by relevance threshold
            # ChromaDB cosine distance: 0 = identical, 1 = opposite
            filtered_chunks = []
            for doc, dist, meta in zip(documents, distances, metadatas):
                if dist <= (1.0 - min_relevance_threshold):
                    filtered_chunks.append(doc)
                    logger.debug(
                        f"RAG hit | condition={meta.get('condition')} "
                        f"| section={meta.get('section')} "
                        f"| distance={dist:.3f}"
                    )
                else:
                    logger.debug(f"RAG chunk filtered out | distance={dist:.3f}")

            logger.info(
                f"RAG retrieval | query={query[:50]!r} | "
                f"retrieved={len(filtered_chunks)}/{top_k}"
            )
            return filtered_chunks

        except Exception as exc:
            logger.error(f"RAG retrieval failed: {exc}")
            raise RAGError(message=str(exc)) from exc

    async def retrieve_for_triage(self, symptoms_text: str) -> List[str]:
        """
        Convenience method: retrieve broader context (top 4) for triage assessment.
        Uses a slightly lower threshold to cast a wider net.
        """
        return await self.retrieve(
            query=symptoms_text,
            top_k=4,
            min_relevance_threshold=0.25,
        )

    async def retrieve_triage_guidance(self, condition_name: str) -> List[str]:
        """
        Retrieve the triage guidance section for a specific diagnosed condition.
        Used after triage scoring to get detailed guidance text.
        """
        return await self.retrieve(
            query=f"triage guidance {condition_name}",
            top_k=2,
            condition_filter=condition_name,
            section_filter="Triage guidance",
        )
