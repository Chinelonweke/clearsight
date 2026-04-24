from __future__ import annotations
"""
app/services/analytics_service.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Built-in analytics and metrics service.

Uses a local SQLite database (data/metrics.db) â€” zero cost,
zero external services, queryable with standard SQL.

Event types tracked:
  session_start       â€” new conversation started
  session_end         â€” conversation completed
  triage_complete     â€” triage scoring finished
  booking_created     â€” appointment successfully booked
  booking_failed      â€” no slots available
  stt_complete        â€” speech transcribed
  tts_complete        â€” speech synthesised
  vision_analysis     â€” eye image analysed
  rag_retrieval       â€” knowledge base queried
  intake_complete     â€” intake form filled
  auth_login          â€” admin login
  api_error           â€” endpoint returned 4xx/5xx

Dashboard data is served from this DB via /admin/dashboard.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_DB_PATH = settings.metrics_db_path


def init_metrics_db() -> None:
    """
    Create the metrics SQLite database and tables if they don't exist.
    Called once at app startup â€” safe to call multiple times (idempotent).
    """
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type  TEXT    NOT NULL,
            session_id  TEXT,
            patient_id  TEXT,
            duration_ms INTEGER,
            success     INTEGER DEFAULT 1,   -- 1 = success, 0 = failure
            metadata    TEXT    DEFAULT '{}',
            created_at  TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON events(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON events(session_id)")
    conn.commit()
    conn.close()
    logger.success(f"Metrics DB initialised | path={_DB_PATH}")


# â”€â”€ Write â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _write_event_sync(
    event_type: str,
    session_id: str | None,
    patient_id: str | None,
    duration_ms: int | None,
    success: bool,
    metadata: dict,
) -> None:
    """Synchronous SQLite write — called in thread executor."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        conn.execute(
            """INSERT INTO events
            (event_type, session_id, patient_id, duration_ms, success, metadata)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                event_type,
                session_id,
                patient_id,
                duration_ms,
                1 if success else 0,
                json.dumps(metadata),
            ),
        )
        conn.commit()
    finally:
        conn.close()


async def track_event(
    event_type: str,
    session_id: str | None = None,
    patient_id: str | None = None,
    duration_ms: int | None = None,
    success: bool = True,
    metadata: dict | None = None,
) -> None:
    """
    Track an analytics event asynchronously.
    Runs the SQLite write in a thread executor to avoid blocking the event loop.

    Usage:
        await track_event("triage_complete", session_id="abc", duration_ms=1200,
                          metadata={"urgency": "urgent", "score": 7})
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        _write_event_sync,
        event_type,
        session_id,
        patient_id,
        duration_ms,
        success,
        metadata or {},
    )
    logger.debug(
        f"Event tracked | type={event_type} | session={session_id} | "
        f"success={success} | duration={duration_ms}ms"
    )


# â”€â”€ Dashboard Queries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_dashboard_stats() -> dict:
    """
    Query the metrics DB for dashboard data.
    Returns a dict consumed by the /admin/dashboard HTML template.
    Called synchronously (in a route handler, use run_in_executor if needed).
    """
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        # â”€â”€ Totals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        total_sessions = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='session_start'"
        ).fetchone()[0]

        sessions_today = conn.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE event_type='session_start' AND date(created_at)=date('now')"
        ).fetchone()[0]

        bookings_today = conn.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE event_type='booking_created' AND date(created_at)=date('now')"
        ).fetchone()[0]

        bookings_total = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='booking_created'"
        ).fetchone()[0]

        booking_failures_today = conn.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE event_type='booking_failed' AND date(created_at)=date('now')"
        ).fetchone()[0]

        # â”€â”€ Urgency breakdown (all time) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        urgency_rows = conn.execute(
            "SELECT json_extract(metadata,'$.urgency_level') as level, COUNT(*) as cnt "
            "FROM events WHERE event_type='triage_complete' "
            "GROUP BY level"
        ).fetchall()
        urgency_breakdown = {row["level"]: row["cnt"] for row in urgency_rows if row["level"]}

        # â”€â”€ Average response times (ms) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        avg_triage_ms = conn.execute(
            "SELECT ROUND(AVG(duration_ms),0) FROM events "
            "WHERE event_type='triage_complete' AND duration_ms IS NOT NULL"
        ).fetchone()[0] or 0

        avg_stt_ms = conn.execute(
            "SELECT ROUND(AVG(duration_ms),0) FROM events "
            "WHERE event_type='stt_complete' AND duration_ms IS NOT NULL"
        ).fetchone()[0] or 0

        # â”€â”€ Hourly session distribution (today) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        hourly_rows = conn.execute(
            "SELECT strftime('%H', created_at) as hour, COUNT(*) as cnt "
            "FROM events WHERE event_type='session_start' "
            "AND date(created_at)=date('now') "
            "GROUP BY hour ORDER BY hour"
        ).fetchall()
        hourly_sessions = {row["hour"]: row["cnt"] for row in hourly_rows}
        # Fill gaps with 0
        hourly_sessions_full = {f"{h:02d}": hourly_sessions.get(f"{h:02d}", 0) for h in range(8, 19)}

        # â”€â”€ Recent events (last 20) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        recent_rows = conn.execute(
            "SELECT event_type, session_id, success, created_at "
            "FROM events ORDER BY id DESC LIMIT 20"
        ).fetchall()
        recent_events = [dict(row) for row in recent_rows]

        # â”€â”€ Error rate today â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        errors_today = conn.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE success=0 AND date(created_at)=date('now')"
        ).fetchone()[0]

        total_today = conn.execute(
            "SELECT COUNT(*) FROM events WHERE date(created_at)=date('now')"
        ).fetchone()[0]

        error_rate = round((errors_today / total_today * 100), 1) if total_today > 0 else 0.0

        return {
            "total_sessions": total_sessions,
            "sessions_today": sessions_today,
            "bookings_today": bookings_today,
            "bookings_total": bookings_total,
            "booking_failures_today": booking_failures_today,
            "urgency_breakdown": urgency_breakdown,
            "avg_triage_ms": int(avg_triage_ms),
            "avg_stt_ms": int(avg_stt_ms),
            "hourly_sessions": hourly_sessions_full,
            "recent_events": recent_events,
            "error_rate_today": error_rate,
            "errors_today": errors_today,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    finally:
        conn.close()
