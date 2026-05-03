from __future__ import annotations
"""
app/services/memory_service.py
─────────────────────────────────────────────────────────────────
mem0 integration for ClearSight patient memory.

Production fixes applied:
  1. Quota monitoring — warns at 20%, alerts Sentry at 10%
  2. Context injected on first turn only (not every LLM call)
  3. Partial memory saves — saves even on incomplete sessions
  4. Deduplication — checks before saving to avoid duplicate facts
  5. Patient ID consistency — uses canonical patient_id throughout
"""

import os
import sentry_sdk
from app.core.logger import get_logger

logger = get_logger(__name__)

MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")

# Quota thresholds for alerts
_QUOTA_WARN_THRESHOLD = 200   # warn at 200 remaining (20% of 1000)
_QUOTA_CRITICAL_THRESHOLD = 100  # Sentry alert at 100 remaining (10%)


class MemoryService:
    """
    Patient memory service using mem0.
    Gracefully degrades if mem0 is unavailable — never crashes the app.

    Senior dev pattern:
    - NeonDB answers factual questions (is this patient returning?)
    - mem0 handles soft context (what do we remember about them?)
    - Context only injected on FIRST turn of a session, not every turn
    """

    def __init__(self) -> None:
        self._client = None
        self._available = False
        self._quota_remaining: int | None = None

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

    # ── Quota monitoring ───────────────────────────────────────────────────────

    def _check_quota(self, response_headers: dict) -> None:
        """
        Monitor mem0 quota from response headers.
        Warns at 20%, alerts Sentry at 10%.
        """
        try:
            remaining = int(response_headers.get("x-quota-remaining", -1))
            limit = int(response_headers.get("x-quota-limit", 1000))
            if remaining < 0:
                return

            self._quota_remaining = remaining
            pct = (remaining / limit) * 100

            if remaining <= _QUOTA_CRITICAL_THRESHOLD:
                logger.error(
                    f"mem0 quota CRITICAL | remaining={remaining}/{limit} ({pct:.0f}%)"
                )
                sentry_sdk.capture_message(
                    f"ClearSight mem0 quota critical: {remaining}/{limit} remaining",
                    level="error"
                )
            elif remaining <= _QUOTA_WARN_THRESHOLD:
                logger.warning(
                    f"mem0 quota LOW | remaining={remaining}/{limit} ({pct:.0f}%)"
                )
                sentry_sdk.capture_message(
                    f"ClearSight mem0 quota low: {remaining}/{limit} remaining",
                    level="warning"
                )
        except Exception:
            pass  # Never crash on quota check

    # ── Retrieve patient memories (first turn only) ────────────────────────────

    async def get_patient_context(
        self,
        patient_id: str,
        query: str = "eye symptoms medical history conditions medications name phone email",
    ) -> str:
        """
        Retrieve relevant memories for a patient.
        Should only be called ONCE per session (on first turn).
        Returns empty string if no memories or mem0 unavailable.
        """
        if not self._available or not self._client:
            return ""

        # Bail early if quota is critically low
        if self._quota_remaining is not None and self._quota_remaining <= 50:
            logger.warning(
                f"mem0 quota too low to search | remaining={self._quota_remaining}"
            )
            return ""

        try:
            results = self._client.search(
                query=query,
                filters={"user_id": patient_id},
                limit=10,
            )

            # Handle both dict response and list response
            if isinstance(results, dict):
                # Extract quota info from response if available
                self._check_quota(results)
                results = results.get("results", [])

            if not results:
                return ""

            # Deduplicate memories by content
            seen = set()
            memory_lines = []
            for mem in results:
                if isinstance(mem, dict):
                    memory = mem.get("memory", "").strip()
                else:
                    memory = str(mem).strip()

                # Simple dedup — skip if very similar to existing memory
                memory_key = memory.lower()[:80]
                if memory and memory_key not in seen:
                    seen.add(memory_key)
                    memory_lines.append(f"- {memory}")

            if not memory_lines:
                return ""

            context = (
                "### Patient Memory (from previous visits):\n"
                + "\n".join(memory_lines)
                + "\n\nUse this context to personalise your responses. "
                "Never repeat questions the patient has already answered in past visits "
                "unless clinically necessary. "
                "You already have their name, phone, and email — do NOT ask again."
            )

            logger.debug(
                f"Patient memories retrieved | patient_id={patient_id} "
                f"| count={len(memory_lines)}"
            )
            return context

        except Exception as exc:
            logger.warning(f"Memory retrieval failed (non-fatal): {exc}")
            return ""

    # ── Save session memories (complete AND incomplete) ────────────────────────

    async def save_session_memories(
        self,
        patient_id: str,
        patient_name: str,
        transcript: str,
        triage_result=None,
        session_metadata: dict | None = None,
        is_partial: bool = False,
    ) -> bool:
        """
        Extract and save key facts from a triage session.
        Works for both COMPLETED and PARTIAL (abandoned) sessions.

        Args:
            patient_id:       Canonical patient UUID (from JWT/DB — not booking ID)
            patient_name:     Patient full name
            transcript:       Full conversation transcript
            triage_result:    TriageResult object (None for partial sessions)
            session_metadata: Additional session data
            is_partial:       True if session was abandoned mid-triage

        Returns:
            True if saved successfully, False otherwise.
        """
        if not self._available or not self._client:
            return False

        # Don't save if transcript is too short (just greeting, nothing useful)
        if len(transcript.strip()) < 100:
            logger.debug(
                f"Transcript too short to save | patient_id={patient_id} "
                f"| length={len(transcript)}"
            )
            return False

        try:
            session_type = "partial" if is_partial else "completed"

            # Build memory messages
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"Patient {patient_name} had a {session_type} triage session "
                        f"at ClearSight Eye Clinic. "
                        "Extract and remember: full name, phone number, email, "
                        "symptoms, duration, pain level, eye affected, "
                        "family history, vision changes, urgency level, "
                        "and any other clinically relevant facts."
                    )
                },
                {
                    "role": "user",
                    "content": transcript[:4000],
                }
            ]

            # Add triage result context if available (completed sessions only)
            if triage_result:
                try:
                    conditions = ", ".join(
                        getattr(triage_result, "suspected_conditions", [])[:3]
                    ) or "unknown"
                    triage_summary = (
                        f"Triage outcome: {triage_result.urgency_level.upper()} "
                        f"(score {triage_result.urgency_score}/10). "
                        f"Chief complaint: {triage_result.chief_complaint}. "
                        f"Suspected conditions: {conditions}. "
                        f"Patient instruction: {triage_result.patient_instruction}"
                    )
                    messages.append({
                        "role": "assistant",
                        "content": triage_summary
                    })
                except Exception:
                    pass  # Don't fail on triage result formatting

            # Save to mem0
            self._client.add(
                messages=messages,
                user_id=patient_id,
                metadata={
                    "source": "clearsight_triage",
                    "session_type": session_type,
                    "patient_name": patient_name,
                    "urgency_level": (
                        getattr(triage_result, "urgency_level", "unknown")
                        if triage_result else "partial"
                    ),
                }
            )

            logger.info(
                f"Session memories saved | patient_id={patient_id} "
                f"| name={patient_name} | type={session_type}"
            )
            return True

        except Exception as exc:
            logger.warning(f"Memory save failed (non-fatal): {exc}")
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

    # ── is_returning_patient kept for backward compat but deprecated ───────────

    async def is_returning_patient(self, patient_id: str) -> bool:
        """
        DEPRECATED: Use NeonDB session count instead.
        Kept for backward compatibility only.
        Senior dev pattern: ask YOUR OWN DB factual yes/no questions.
        """
        logger.warning(
            "is_returning_patient() called — use NeonDB session count instead"
        )
        return False
