from __future__ import annotations
"""
app/services/intake_service.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Intake form auto-fill service.

After the triage conversation ends, this service:
  1. Sends the full conversation transcript to LLaMA3 with a structured
     extraction prompt.
  2. Parses the structured output into an IntakeForm ORM record.
  3. Persists the form to NeonDB.

The auto-filled fields match the IntakeForm model:
  - symptoms (dict)
  - vision_changes, pain_level, duration, eye_affected
  - raw_transcript, ai_summary
  - image_urls (passed in from vision analysis)
"""

import json
import uuid as _uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import LLMError
from app.core.logger import get_logger
from app.models.intake import IntakeForm
from app.services.llm_service import LLMService

logger = get_logger(__name__)

_INTAKE_EXTRACTION_SYSTEM = """You are a medical data extraction assistant for a Nigerian eye clinic.
Extract structured patient intake information from a triage conversation transcript.

Return ONLY a valid JSON object with these exact keys:
{
  "full_name": "<patient's full name or null>",
  "phone": "<phone number or null>",
  "date_of_birth": "<YYYY-MM-DD or null>",
  "symptoms": {
    "primary": "<main symptom>",
    "secondary": ["<symptom 2>", "<symptom 3>"],
    "onset": "<when symptoms started>",
    "character": "<how the symptom feels: sharp, burning, itchy, etc.>"
  },
  "vision_changes": "<describe any vision changes mentioned or null>",
  "pain_level": <integer 0â€“10 or null>,
  "duration": "<e.g. '3 days', '2 weeks', 'since this morning'>",
  "eye_affected": "left" | "right" | "both" | null,
  "relevant_history": "<known conditions, medications, prior eye problems or null>",
  "ai_summary": "<2â€“3 sentence clinical summary of the case>"
}

Extract only what is explicitly stated. Use null for missing fields. Never invent information."""


class IntakeService:
    """
    Auto-fills patient intake forms from conversation transcripts.
    """

    def __init__(self, llm: LLMService, db: AsyncSession) -> None:
        self._llm = llm
        self._db = db
        logger.debug("IntakeService initialised.")

    async def extract_intake_data(self, transcript: str) -> dict:
        """
        Run LLM extraction on a conversation transcript.

        Returns:
            Extracted intake data as a dict.
        """
        if not transcript.strip():
            logger.warning("Empty transcript passed to intake extraction.")
            return {}

        logger.info(f"Extracting intake data | transcript_length={len(transcript)}")

        try:
            extracted = await self._llm.complete_json(
                system=_INTAKE_EXTRACTION_SYSTEM,
                user=f"CONVERSATION TRANSCRIPT:\n{transcript}",
                temperature=0.0,    # fully deterministic â€” we want exact extraction
            )
            logger.success(
                f"Intake extraction complete | "
                f"fields={[k for k, v in extracted.items() if v is not None]}"
            )
            return extracted

        except LLMError as exc:
            logger.error(f"Intake extraction LLM call failed: {exc}")
            return {}

    async def create_intake_form(
        self,
        session_id: str,
        patient_id: str | None,
        transcript: str,
        image_urls: list[str] | None = None,
        preloaded_symptoms: dict | None = None,
    ) -> IntakeForm:
        """
        Extract intake data and persist an IntakeForm record to NeonDB.

        Args:
            session_id:         The conversation session UUID string.
            patient_id:         Patient UUID string (may be None for walk-ins).
            transcript:         Full plain-text conversation transcript.
            image_urls:         List of saved eye image paths from VisionService.
            preloaded_symptoms: If symptoms were collected incrementally in Redis,
                                pass them here to merge with LLM extraction.

        Returns:
            Persisted IntakeForm ORM object.
        """
        extracted = await self.extract_intake_data(transcript)

        # Merge pre-collected symptoms from Redis with LLM extraction
        base_symptoms = extracted.get("symptoms") or {}
        if preloaded_symptoms:
            base_symptoms.update(preloaded_symptoms)

        # Parse pain level safely
        pain_raw = extracted.get("pain_level")
        try:
            pain_level = int(pain_raw) if pain_raw is not None else None
            pain_level = max(0, min(10, pain_level)) if pain_level is not None else None
        except (ValueError, TypeError):
            pain_level = None

        # Build and persist the form
        intake_form = IntakeForm(
            patient_id=_uuid.UUID(patient_id) if patient_id else None,
            session_id=_uuid.UUID(session_id) if session_id else None,
            symptoms=base_symptoms,
            vision_changes=extracted.get("vision_changes"),
            pain_level=pain_level,
            duration=extracted.get("duration"),
            eye_affected=extracted.get("eye_affected"),
            image_urls=image_urls or [],
            raw_transcript=transcript,
            ai_summary=extracted.get("ai_summary"),
        )

        self._db.add(intake_form)
        await self._db.flush()

        logger.success(
            f"Intake form created | id={intake_form.id} | "
            f"session={session_id} | patient={patient_id}"
        )
        return intake_form

    async def get_intake_form(self, session_id: str) -> IntakeForm | None:
        """Retrieve the intake form for a given session."""
        result = await self._db.execute(
            select(IntakeForm).where(
                IntakeForm.session_id == _uuid.UUID(session_id)
            )
        )
        return result.scalar_one_or_none()
