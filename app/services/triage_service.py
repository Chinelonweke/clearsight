from __future__ import annotations
"""
app/services/triage_service.py
Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
Clinical triage engine Ã¢â‚¬â€ scores patient urgency from 1Ã¢â‚¬â€œ10
and determines the recommended care pathway.

Flow:
  1. Receive full symptom transcript + session metadata.
  2. Retrieve relevant clinical context from RAG (ChromaDB).
  3. Pass everything to LLaMA3 with a low-temperature structured prompt.
  4. Parse and validate the JSON triage response.
  5. Return a TriageResult with urgency level, score, and booking guidance.

Urgency levels:
  EMERGENCY (9Ã¢â‚¬â€œ10) : Same-day, prioritised, doctor notified immediately.
  URGENT    (6Ã¢â‚¬â€œ8)  : Same-day or next available slot today.
  ROUTINE   (1Ã¢â‚¬â€œ5)  : Standard scheduling, next available slot this week.
"""

import json

from app.core.exceptions import LLMError
from app.core.logger import get_logger
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService

logger = get_logger(__name__)

# Emergency keywords for fast-path detection
EMERGENCY_KEYWORDS = [
    "chemical", "acid", "lime", "bleach", "cement", "explosion",
    "curtain", "shadow across vision", "sudden total blindness",
    "penetrating", "stabbed", "nail", "metal in eye",
]

# Ã¢â€â‚¬Ã¢â€â‚¬ Triage system prompt Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
# Deliberately low temperature (0.05) Ã¢â‚¬â€ we need consistent, reproducible scoring.
_TRIAGE_SYSTEM = """You are a clinical triage nurse at a Nigerian optometrist eye clinic.
You assess patient-reported eye symptoms and assign an urgency level.

URGENCY LEVELS Ã¢â‚¬â€ use these definitions exactly:
- EMERGENCY (score 9Ã¢â‚¬â€œ10): Conditions that may cause permanent blindness within hours if untreated.
  Examples: sudden vision loss, chemical splash, retinal detachment signs (curtain/shadow across vision),
  acute angle-closure glaucoma (severe pain + halos + nausea), penetrating eye injury, vitreous haemorrhage.
- URGENT (score 6Ã¢â‚¬â€œ8): Conditions requiring assessment within 24Ã¢â‚¬â€œ48 hours.
  Examples: significant eye pain, sudden increase in floaters/flashes, corneal ulcer suspected,
  uveitis symptoms, severe conjunctivitis not responding to treatment, vision significantly reduced.
- ROUTINE (score 1Ã¢â‚¬â€œ5): Conditions suitable for standard scheduling within the week.
  Examples: gradual blur, reading difficulty, dry eyes, mild irritation, routine check-up,
  known stable conditions needing monitoring.

IMPORTANT RULES:
1. When in doubt between two levels, always escalate to the higher one.
2. Chemical splash to the eye is ALWAYS score 10, regardless of current symptoms.
3. Never downgrade emergency symptoms based on patient reassurance.
4. Consider the Nigerian clinical context: patients often present late; symptoms may be underreported.

You MUST respond ONLY with a valid JSON object. No prose before or after the JSON.

Required JSON structure:
{
  "urgency_level": "emergency" | "urgent" | "routine",
  "urgency_score": <integer 1Ã¢â‚¬â€œ10>,
  "chief_complaint": "<one sentence summary of the main problem>",
  "suspected_conditions": ["<condition 1>", "<condition 2>"],
  "recommended_timeframe": "immediate" | "same-day" | "within-48h" | "within-week" | "routine",
  "triage_reasoning": "<2Ã¢â‚¬â€œ3 sentence clinical reasoning>",
  "red_flags_detected": ["<flag 1>", "<flag 2>"],
  "suggested_questions": ["<follow-up question 1>", "<follow-up question 2>"],
  "patient_instruction": "<short instruction to give the patient now, in plain English>"
}"""


class TriageResult:
    """Structured output from a triage assessment."""

    def __init__(self, data: dict) -> None:
        self.urgency_level: str = data.get("urgency_level", "routine").lower()
        self.urgency_score: int = int(data.get("urgency_score", 3))
        self.chief_complaint: str = data.get("chief_complaint", "")
        self.suspected_conditions: list[str] = data.get("suspected_conditions", [])
        self.recommended_timeframe: str = data.get("recommended_timeframe", "within-week")
        self.triage_reasoning: str = data.get("triage_reasoning", "")
        self.red_flags_detected: list[str] = data.get("red_flags_detected", [])
        self.suggested_questions: list[str] = data.get("suggested_questions", [])
        self.patient_instruction: str = data.get("patient_instruction", "")
        self.raw: dict = data

    @property
    def is_emergency(self) -> bool:
        return self.urgency_level == "emergency"

    @property
    def is_routine(self) -> bool:
        return self.urgency_level.lower() == "routine"

    @property
    def is_urgent(self) -> bool:
        return self.urgency_level == "urgent"

    def to_dict(self) -> dict:
        return {
            "urgency_level": self.urgency_level,
            "urgency_score": self.urgency_score,
            "chief_complaint": self.chief_complaint,
            "suspected_conditions": self.suspected_conditions,
            "recommended_timeframe": self.recommended_timeframe,
            "triage_reasoning": self.triage_reasoning,
            "red_flags_detected": self.red_flags_detected,
            "suggested_questions": self.suggested_questions,
            "patient_instruction": self.patient_instruction,
        }

    def __repr__(self) -> str:
        return (
            f"<TriageResult level={self.urgency_level} "
            f"score={self.urgency_score} "
            f"complaint={self.chief_complaint[:40]!r}>"
        )


class TriageService:
    """
    Urgency scoring engine combining LLaMA3 reasoning with RAG-retrieved
    clinical knowledge.
    """

    def __init__(self, llm: LLMService, rag: RAGService) -> None:
        self._llm = llm
        self._rag = rag
        logger.info("TriageService initialised.")

    async def assess(
        self,
        symptoms_transcript: str,
        patient_metadata: dict | None = None,
        vision_observation: str | None = None,
    ) -> TriageResult:
        """
        Perform a full triage assessment.

        Args:
            symptoms_transcript: Full text of what the patient described.
            patient_metadata:    Dict with patient context Ã¢â‚¬â€ age, known conditions, etc.
            vision_observation:  Optional output from VisionService image analysis.

        Returns:
            TriageResult with urgency level, score, and guidance.
        """
        logger.info(
            f"Triage assessment starting | "
            f"transcript_length={len(symptoms_transcript)} | "
            f"has_image={vision_observation is not None}"
        )

        # Ã¢â€â‚¬Ã¢â€â‚¬ RAG: retrieve relevant clinical context Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        context_chunks = []
        try:
            context_chunks = await self._rag.retrieve_for_triage(symptoms_transcript)
        except Exception as exc:
            logger.warning(f"RAG retrieval failed Ã¢â‚¬â€ proceeding without context: {exc}")

        # Ã¢â€â‚¬Ã¢â€â‚¬ Build the user message Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        user_parts = []

        if patient_metadata:
            user_parts.append(f"PATIENT INFORMATION:\n{json.dumps(patient_metadata, indent=2)}")

        user_parts.append(f"REPORTED SYMPTOMS (conversation transcript):\n{symptoms_transcript}")

        if vision_observation:
            user_parts.append(f"EYE IMAGE OBSERVATION (AI visual analysis):\n{vision_observation}")

        if context_chunks:
            context_block = "\n\n---\n".join(context_chunks)
            user_parts.append(
                f"CLINICAL REFERENCE (retrieved from knowledge base):\n{context_block}"
            )

        user_parts.append(
            "Based on all the above, perform a clinical triage assessment and "
            "return the JSON response as specified."
        )

        user_message = "\n\n".join(user_parts)

        # Ã¢â€â‚¬Ã¢â€â‚¬ LLM call Ã¢â‚¬â€ JSON mode Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        try:
            triage_data = await self._llm.complete_json(
                system=_TRIAGE_SYSTEM,
                user=user_message,
                temperature=0.05,   # near-deterministic for consistent clinical scoring
            )
        except LLMError as exc:
            logger.error(f"LLM triage call failed: {exc}")
            raise

        # Ã¢â€â‚¬Ã¢â€â‚¬ Validate and clamp score Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        score = int(triage_data.get("urgency_score", 3))
        triage_data["urgency_score"] = max(1, min(10, score))

    # Emergency override: if any red-flag keyword detected, force minimum score
    transcript_lower = symptoms_transcript.lower()
    for kw in EMERGENCY_KEYWORDS:
            if kw in transcript_lower:
                if triage_data["urgency_score"] < 9:
                    logger.warning(
                        f"Emergency keyword '{kw}' detected Ã¢â‚¬â€ "
                        f"overriding score {triage_data['urgency_score']} Ã¢â€ â€™ 9"
                    )
                    triage_data["urgency_score"] = 9
                    triage_data["urgency_level"] = "emergency"
                break

        result = TriageResult(triage_data)

        logger.info(
            f"Triage complete | level={result.urgency_level} | "
            f"score={result.urgency_score} | "
            f"complaint={result.chief_complaint[:60]!r}"
        )

        if result.is_emergency:
            logger.warning(
                f"EMERGENCY TRIAGE | score={result.urgency_score} | "
                f"flags={result.red_flags_detected}"
            )

        return result

