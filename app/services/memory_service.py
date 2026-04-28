from __future__ import annotations
"""
app/services/memory_service.py
─────────────────────────────────────────────────────────────────
mem0 integration for ClearSight patient memory.

What it does:
  - Remembers returning patients across sessions
  - Stores key facts: symptoms, conditions, preferences, visit history
  - Injects memories into LLM context for personalised triage
  - Saves new facts after each completed session

Usage:
    memory = MemoryService()

    # Get memories before session
    context = await memory.get_patient_context(patient_id="abc123")

    # Save memories after session
    await memory.save_session_memories(
        patient_id="abc123",
        patient_name="Chinelo",
        transcript="...",
        triage_result=result,
    )

Free tier: mem0 cloud — unlimited memories for development.
Get key at: app.mem0.ai
"""

import os
from app.core.logger import get_logger

logger = get_logger(__name__)

MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")


class MemoryService:
    """
    Patient memory service using mem0.
    Gracefully degrades if mem0 is unavailable — never crashes the app.
    """

    def __init__(self) -> None:
        self._client = None
        self._available = False

        if not MEM0_API_KEY:
            logger.warning("MEM0_API_KEY not set — memory features disabled")
            return

        try:
            from mem0 import MemoryClient
            self._client = MemoryClient(api_key=MEM0_API_KEY)
            self._available = True
            logger.info("MemoryService initialised | provider=mem0 cloud")
        except ImportError:
            logger.warning("mem0 package not installed — run: pip install mem0ai")
        except Exception as exc:
            logger.warning(f"MemoryService init failed: {exc}")

    # ── Retrieve patient memories ──────────────────────────────────────────────

    async def get_patient_context(
        self,
        patient_id: str,
        query: str = "eye symptoms medical history conditions medications",
    ) -> str:
        """
        Retrieve relevant memories for a patient and format as context string.
        Returns empty string if no memories or mem0 unavailable.

        Args:
            patient_id: Unique patient identifier (UUID from DB)
            query:      What to search for in memories

        Returns:
            Formatted string of memories to inject into LLM system prompt.
        """
        if not self._available or not self._client:
            return ""

        try:
            results = self._client.search(
                query=query,
                filters={"user_id": patient_id},
                limit=8,
            )

            if not results:
                return ""

            # Format memories into a readable context block
            memory_lines = []
            for mem in results:
                if isinstance(mem, dict):
                    memory = mem.get("memory", "")
                else:
                    memory = str(mem)
                if memory:
                    memory_lines.append(f"- {memory}")

            if not memory_lines:
                return ""

            context = (
                "### Patient Memory (from previous visits):\n"
                + "\n".join(memory_lines)
                + "\n\nUse this context to personalise your responses. "
                "Acknowledge returning patients warmly. "
                "Never repeat questions the patient has already answered in past visits "
                "unless clinically necessary."
            )

            logger.debug(
                f"Patient memories retrieved | patient_id={patient_id} "
                f"| count={len(memory_lines)}"
            )
            return context

        except Exception as exc:
            logger.warning(f"Memory retrieval failed (non-fatal): {exc}")
            return ""

    # ── Save session memories ──────────────────────────────────────────────────

    async def save_session_memories(
        self,
        patient_id: str,
        patient_name: str,
        transcript: str,
        triage_result=None,
        session_metadata: dict | None = None,
    ) -> bool:
        """
        Extract and save key facts from a completed triage session.

        Args:
            patient_id:       Patient UUID
            patient_name:     Patient full name
            transcript:       Full conversation transcript
            triage_result:    TriageResult object (optional)
            session_metadata: Additional session data (optional)

        Returns:
            True if saved successfully, False otherwise.
        """
        if not self._available or not self._client:
            return False

        try:
            # Build memory messages from transcript
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"Patient {patient_name} visited ClearSight Eye Clinic. "
                        "Extract and remember: symptoms, duration, pain level, "
                        "eye affected, family history, vision changes, urgency level, "
                        "and any other clinically relevant facts."
                    )
                },
                {
                    "role": "user",
                    "content": transcript[:3000],  # Limit transcript length
                }
            ]

            # Add triage result as additional context
            if triage_result:
                triage_summary = (
                    f"Triage outcome: {triage_result.urgency_level.upper()} "
                    f"(score {triage_result.urgency_score}/10). "
                    f"Chief complaint: {triage_result.chief_complaint}. "
                    f"Suspected conditions: {', '.join(triage_result.suspected_conditions[:3])}. "
                    f"Patient instruction: {triage_result.patient_instruction}"
                )
                messages.append({
                    "role": "assistant",
                    "content": triage_summary
                })

            # Save to mem0
            self._client.add(
                messages=messages,
                user_id=patient_id,
                metadata={
                    "source": "clearsight_triage",
                    "patient_name": patient_name,
                    "urgency_level": triage_result.urgency_level if triage_result else "unknown",
                }
            )

            logger.info(
                f"Session memories saved | patient_id={patient_id} "
                f"| name={patient_name}"
            )
            return True

        except Exception as exc:
            logger.warning(f"Memory save failed (non-fatal): {exc}")
            return False

    # ── Check if returning patient ─────────────────────────────────────────────

    async def is_returning_patient(self, patient_id: str) -> bool:
        """Check if patient has any previous memories."""
        if not self._available or not self._client:
            return False

        try:
            results = self._client.search(
                query="patient visit history",
                filters={"user_id": patient_id},
                limit=1,
            )
            return bool(results)
        except Exception:
            return False

    # ── Delete patient memories (GDPR/privacy) ─────────────────────────────────

    async def delete_patient_memories(self, patient_id: str) -> bool:
        """Delete all memories for a patient. For GDPR compliance."""
        if not self._available or not self._client:
            return False

        try:
            self._client.delete_all(user_id=patient_id)
            logger.info(f"Patient memories deleted | patient_id={patient_id}")
            return True
        except Exception as exc:
            logger.warning(f"Memory deletion failed: {exc}")
            return False


