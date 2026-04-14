from __future__ import annotations
"""
app/api/v1/triage.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Triage REST endpoint (for direct API use, not WebSocket).

POST /api/v1/triage/assess â€” assess symptoms and return urgency score
"""

import time

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.logger import get_logger
from app.services.analytics_service import track_event
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.triage_service import TriageService

logger = get_logger(__name__)
router = APIRouter()


class TriageRequest(BaseModel):
    symptoms_text: str
    session_id: str | None = None
    patient_metadata: dict | None = None
    vision_observation: str | None = None


@router.post("/assess")
async def assess_triage(body: TriageRequest):
    """
    Run a triage assessment from a symptoms description.
    Returns urgency level, score, suspected conditions, and patient instructions.
    """
    t_start = time.perf_counter()

    llm = LLMService()
    rag = RAGService()
    svc = TriageService(llm=llm, rag=rag)

    result = await svc.assess(
        symptoms_transcript=body.symptoms_text,
        patient_metadata=body.patient_metadata,
        vision_observation=body.vision_observation,
    )

    duration_ms = int((time.perf_counter() - t_start) * 1000)
    await track_event(
        "triage_complete",
        session_id=body.session_id,
        duration_ms=duration_ms,
        metadata=result.to_dict(),
    )

    return {"triage": result.to_dict(), "duration_ms": duration_ms}
