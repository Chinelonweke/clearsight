from __future__ import annotations
"""
app/api/v1/session.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Session management REST endpoints.

POST /api/v1/session/start       â€” create a new session, return session_id
GET  /api/v1/session/{id}        â€” get session metadata + stage
GET  /api/v1/session/{id}/history â€” get full chat history
DELETE /api/v1/session/{id}      â€” close/delete a session
"""

from fastapi import APIRouter, Depends, HTTPException
from redis.asyncio import Redis

from app.core.logger import get_logger
from app.db.redis_client import get_redis
from app.services.analytics_service import track_event
from app.services.session_service import SessionService

logger = get_logger(__name__)
router = APIRouter()


@router.post("/start")
async def start_session():
    """Create a new triage conversation session."""
    import uuid
    session_id = str(uuid.uuid4())
    
    # Try Redis — if unavailable, return session ID anyway
    try:
        from app.db.redis_client import get_redis_client
        redis = get_redis_client()
        svc = SessionService(redis)
        await svc.create_session()
        await svc.update_metadata(session_id, {"session_id": session_id})
    except Exception:
        pass  # Redis unavailable — session still works for WebSocket
    
    await track_event("session_start", session_id=session_id)
    return {"session_id": session_id, "ws_url": f"/ws/conversation/{session_id}"}


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
