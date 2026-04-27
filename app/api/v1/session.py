from __future__ import annotations
"""
app/api/v1/session.py
─────────────────────────────────────────────────────────
Session management REST endpoints.

POST /api/v1/session/start       — create a new session, return session_id
GET  /api/v1/session/{id}        — get session metadata + stage
GET  /api/v1/session/{id}/history — get full chat history
DELETE /api/v1/session/{id}      — close/delete a session
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from redis.asyncio import Redis

from app.core.logger import get_logger
from app.core.security import verify_token
from app.db.redis_client import get_redis
from app.services.analytics_service import track_event
from app.services.session_service import SessionService

logger = get_logger(__name__)
router = APIRouter()


@router.post("/start")
async def start_session(request: Request):
    """
    Create a new triage conversation session.
    If a valid patient JWT is provided, the session is linked to that patient.
    """
    session_id = str(uuid.uuid4())
    patient_id = None

    # Try to extract patient_id from JWT if provided
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = verify_token(token, expected_type="access")
            if payload.get("role") == "patient":
                patient_id = payload.get("sub")
                logger.info(f"Session linked to patient | patient_id={patient_id} | session_id={session_id}")
        except Exception as exc:
            logger.debug(f"Session started without patient link | reason={exc}")

    # Try Redis — if unavailable, return session ID anyway
    try:
        from app.db.redis_client import get_redis_client
        redis = get_redis_client()
        svc = SessionService(redis)
        await svc.create_session()
        metadata = {"session_id": session_id}
        if patient_id:
            metadata["patient_id"] = patient_id
        await svc.update_metadata(session_id, metadata)
    except Exception:
        pass  # Redis unavailable — session still works for WebSocket

    await track_event("session_start", session_id=session_id)
    return {
        "session_id": session_id,
        "ws_url": f"/ws/conversation/{session_id}",
        "patient_id": patient_id,
    }


@router.get("/{session_id}")
async def get_session(session_id: str, redis: Redis = Depends(get_redis)):
    """Get session metadata and current stage."""
    svc = SessionService(redis)
    if not await svc.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    meta = await svc.get_metadata(session_id)
    return meta


@router.get("/{session_id}/history")
async def get_history(session_id: str, redis: Redis = Depends(get_redis)):
    """Get the full conversation history for a session."""
    svc = SessionService(redis)
    history = await svc.get_history(session_id, include_timestamps=True)
    return {"session_id": session_id, "history": history, "count": len(history)}


@router.delete("/{session_id}")
async def close_session(session_id: str, redis: Redis = Depends(get_redis)):
    """Close a session."""
    svc = SessionService(redis)
    await svc.close_session(session_id, outcome="closed_via_api")
    await track_event("session_end", session_id=session_id)
    return {"status": "closed", "session_id": session_id}