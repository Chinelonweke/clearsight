from __future__ import annotations
"""
app/services/session_service.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Manages per-patient conversation sessions in Redis.

Every WebSocket conversation gets a session_id (UUID).
This service stores:
  - Full chat history  (Redis list,  key: session:{id}:history)
  - Session metadata   (Redis hash,  key: session:{id}:meta)
  - Collected symptoms (Redis hash,  key: session:{id}:symptoms)

All keys expire after SESSION_TTL seconds (6 hours by default).
On session close, the final state is persisted to NeonDB via
ConversationSession + IntakeForm tables.

Usage:
    svc = SessionService(redis_client)
    await svc.append_message(session_id, "user", "My eye is red")
    history = await svc.get_history(session_id)
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from app.core.logger import get_logger

logger = get_logger(__name__)

SESSION_TTL = 6 * 3600          # 6 hours in seconds
MAX_HISTORY_MESSAGES = 100      # cap history to avoid unbounded memory growth


class SessionService:
    """
    Redis-backed session management for ClearSight conversations.
    """

    def __init__(self, redis: Redis) -> None:
        self._r = redis

    # â”€â”€ Key helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"session:{session_id}:history"

    @staticmethod
    def _meta_key(session_id: str) -> str:
        return f"session:{session_id}:meta"

    @staticmethod
    def _symptoms_key(session_id: str) -> str:
        return f"session:{session_id}:symptoms"

    # â”€â”€ Session lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def create_session(
        self,
        session_id: str | None = None,
        patient_id: str | None = None,
        extra_meta: dict | None = None,
    ) -> str:
        """
        Create a new session and store initial metadata.

        Args:
            session_id: Optional caller-supplied UUID string. A new UUID is
                        generated when omitted.

        Returns:
            The session_id used (new or supplied).
        """
        session_id = session_id or str(uuid.uuid4())
        meta: dict[str, Any] = {
            "session_id": session_id,
            "patient_id": patient_id or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": "greeting",        # greeting â†’ collecting â†’ triage â†’ booking â†’ done
            "message_count": 0,
        }
        if extra_meta:
            meta.update(extra_meta)

        await self._r.set(
            self._meta_key(session_id),
            json.dumps(meta),
            ex=SESSION_TTL,
        )
        logger.info(f"Session created | session_id={session_id} | patient={patient_id}")
        return session_id

    async def close_session(self, session_id: str, outcome: str = "completed") -> None:
        """Mark session as closed with a final outcome."""
        meta = await self.get_metadata(session_id)
        meta["ended_at"] = datetime.now(timezone.utc).isoformat()
        meta["outcome"] = outcome
        meta["stage"] = "done"

        await self._r.set(
            self._meta_key(session_id),
            json.dumps(meta),
            ex=SESSION_TTL,
        )
        logger.info(f"Session closed | session_id={session_id} | outcome={outcome}")

    # â”€â”€ Chat history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Append a message to the session history list.

        Args:
            session_id: The conversation session.
            role:       "user" or "assistant".
            content:    The message text.
        """
        key = self._history_key(session_id)
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        pipe = self._r.pipeline()
        pipe.rpush(key, message)
        pipe.ltrim(key, -MAX_HISTORY_MESSAGES, -1)  # keep last N messages only
        pipe.expire(key, SESSION_TTL)
        await pipe.execute()

        logger.debug(
            f"Message appended | session={session_id} | role={role} "
            f"| preview={content[:60]!r}"
        )

        # Increment message counter in metadata
        meta = await self.get_metadata(session_id)
        meta["message_count"] = meta.get("message_count", 0) + 1
        await self._r.set(self._meta_key(session_id), json.dumps(meta), ex=SESSION_TTL)

    async def get_history(
        self,
        session_id: str,
        include_timestamps: bool = False,
    ) -> list[dict]:
        """
        Retrieve full conversation history.

        Returns:
            List of {"role": ..., "content": ...} dicts
            (timestamps included only if include_timestamps=True).
        """
        key = self._history_key(session_id)
        raw_messages = await self._r.lrange(key, 0, -1)

        history = []
        for raw in raw_messages:
            msg = json.loads(raw)
            if include_timestamps:
                history.append(msg)
            else:
                # LLM-ready format â€” only role + content
                history.append({"role": msg["role"], "content": msg["content"]})

        return history

    async def get_history_text(self, session_id: str) -> str:
        """Return the full conversation as a plain-text transcript string."""
        history = await self.get_history(session_id, include_timestamps=True)
        lines = []
        for msg in history:
            ts = msg.get("timestamp", "")[:19].replace("T", " ")
            lines.append(f"[{ts}] {msg['role'].upper()}: {msg['content']}")
        return "\n".join(lines)

    async def clear_history(self, session_id: str) -> None:
        """Delete the history list for this session."""
        await self._r.delete(self._history_key(session_id))
        logger.debug(f"History cleared | session={session_id}")

    # â”€â”€ Session metadata â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def get_metadata(self, session_id: str) -> dict:
        """
        Get session metadata dict.
        Returns an empty dict if the session does not exist.
        """
        raw = await self._r.get(self._meta_key(session_id))
        if not raw:
            return {}
        return json.loads(raw)

    async def update_metadata(self, session_id: str, updates: dict) -> None:
        """Merge updates into existing session metadata."""
        meta = await self.get_metadata(session_id)
        meta.update(updates)
        await self._r.set(self._meta_key(session_id), json.dumps(meta), ex=SESSION_TTL)
        logger.debug(f"Metadata updated | session={session_id} | keys={list(updates.keys())}")

    async def set_stage(self, session_id: str, stage: str) -> None:
        """
        Update the conversation stage.
        Stages: greeting â†’ collecting â†’ triage â†’ booking â†’ done
        """
        await self.update_metadata(session_id, {"stage": stage})
        logger.info(f"Stage updated | session={session_id} | stage={stage}")

    # â”€â”€ Collected symptoms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def store_symptoms(self, session_id: str, symptoms: dict) -> None:
        """
        Store structured symptom data collected during the conversation.
        Merged with any previously stored symptoms.
        """
        key = self._symptoms_key(session_id)
        raw = await self._r.get(key)
        existing = json.loads(raw) if raw else {}
        existing.update(symptoms)
        await self._r.set(key, json.dumps(existing), ex=SESSION_TTL)
        logger.debug(f"Symptoms stored | session={session_id} | fields={list(symptoms.keys())}")

    async def get_symptoms(self, session_id: str) -> dict:
        """Retrieve all collected symptom data for this session."""
        raw = await self._r.get(self._symptoms_key(session_id))
        return json.loads(raw) if raw else {}

    # â”€â”€ Utility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def session_exists(self, session_id: str) -> bool:
        """Return True if the session metadata key exists in Redis."""
        return bool(await self._r.exists(self._meta_key(session_id)))

    async def get_full_snapshot(self, session_id: str) -> dict:
        """
        Return a complete snapshot of the session state.
        Useful for persisting to NeonDB at session close.
        """
        meta = await self.get_metadata(session_id)
        history = await self.get_history(session_id, include_timestamps=True)
        symptoms = await self.get_symptoms(session_id)
        transcript = await self.get_history_text(session_id)

        return {
            "metadata": meta,
            "history": history,
            "symptoms": symptoms,
            "transcript": transcript,
            "message_count": len(history),
        }
