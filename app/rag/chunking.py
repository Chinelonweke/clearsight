from __future__ import annotations
"""
app/rag/chunking.py
─────────────────────────────────────────────────────────
Semantic chunking strategy for the eye condition knowledge base.

Strategy: Sliding-window semantic similarity
  1. Split document into sentences.
  2. Encode each sentence with sentence-transformers (all-MiniLM-L6-v2).
  3. Compute cosine similarity between adjacent sentences.
  4. When similarity drops below threshold -> start a new chunk.
  5. Also enforce a max_sentences guard to prevent runaway chunks.

This outperforms fixed-token chunking for medical text because:
  - Clinical paragraphs have variable lengths.
  - Symptom lists, triage guidance, and risk factors are semantically
    distinct sections -- they should be separate chunks.
  - A query like "symptoms of glaucoma" should retrieve the Symptoms
    section, not a fragment that crosses into Triage guidance.

Typical output: 150-400 tokens per chunk, ~8-20 chunks per condition file.
"""

import re
from typing import List

import numpy as np

from app.core.logger import get_logger

logger = get_logger(__name__)

# Lazy-load to avoid slow import at module level (sentence-transformers takes ~2s)
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.success("Sentence-transformers model loaded.")
    return _embed_model


def _split_sentences(text: str) -> List[str]:
    """
    Split markdown text into sentences, preserving bullet-point items
    as individual units (each bullet = one sentence unit for chunking).
    """
    text = re.sub(r"\r\n", "\n", text)
    sentences: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            sentences.append(line)
            continue
        if line.startswith(("-", "*", ".")):
            sentences.append(line)
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.45,
    max_sentences_per_chunk: int = 12,
    min_chunk_length: int = 80,
) -> List[str]:
    sentences = _split_sentences(text)

    if len(sentences) == 0:
        logger.warning("Document produced zero sentences after splitting.")
        return []

    if len(sentences) == 1:
        return sentences

    model = _get_embed_model()
    embeddings: np.ndarray = model.encode(sentences, show_progress_bar=False)

    chunks: List[str] = []
    current_sentences: List[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        at_max = len(current_sentences) >= max_sentences_per_chunk
        is_heading = sentences[i].startswith("#")

        if sim < similarity_threshold or at_max or is_heading:
            chunk_text = " ".join(current_sentences).strip()
            if len(chunk_text) >= min_chunk_length:
                chunks.append(chunk_text)
            current_sentences = [sentences[i]]
        else:
            current_sentences.append(sentences[i])

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if len(chunk_text) >= min_chunk_length:
            chunks.append(chunk_text)

    logger.debug(
        f"Semantic chunking | sentences={len(sentences)} -> chunks={len(chunks)}"
    )
    return chunks


def chunk_document_with_metadata(
    text: str,
    source_file: str,
    condition_name: str,
) -> List[dict]:
    chunks = semantic_chunk(text)
    results = []
    for i, chunk in enumerate(chunks):
        section = _detect_section(chunk, text)
        results.append({
            "text": chunk,
            "source": source_file,
            "condition": condition_name,
            "chunk_index": i,
            "section": section,
            "char_count": len(chunk),
        })
    return results


def _detect_section(chunk: str, full_text: str) -> str:
    for line in chunk.split():
        if line.startswith("##"):
            return line.lstrip("#").strip()

    keywords = {
        "symptoms": "Symptoms",
        "triage": "Triage guidance",
        "risk factor": "Risk factors",
        "overview": "Overview",
        "type": "Types",
        "treatment": "Treatment",
        "differential": "Differential diagnoses",
        "what the ai": "AI questions",
        "clinical note": "Clinical notes",
    }
    chunk_lower = chunk.lower()
    for kw, label in keywords.items():
        if kw in chunk_lower:
            return label
    return "General"