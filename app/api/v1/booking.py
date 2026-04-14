from __future__ import annotations
"""
app/api/v1/booking.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Booking REST endpoints.

GET  /api/v1/booking/slots         â€” list available slots
POST /api/v1/booking/seed-slots    â€” seed availability for a doctor (admin)
GET  /api/v1/booking/{id}          â€” get appointment details
POST /api/v1/booking/{id}/cancel   â€” cancel an appointment
"""

import uuid
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.logger import get_logger
from app.dependencies import db_session, require_admin
from app.models.booking import Appointment
from app.models.doctor import AvailabilitySlot
from app.services.booking_service import BookingService

logger = get_logger(__name__)
router = APIRouter()


@router.get("/slots")
async def list_slots(
    days_ahead: int = 7,
    doctor_id: str | None = None,
    db: AsyncSession = Depends(db_session),
):
    """List available appointment slots."""
    svc = BookingService(db)
    now = datetime.now(timezone.utc)
    doc_uuid = uuid.UUID(doctor_id) if doctor_id else None

    slots = await svc.get_available_slots(
        from_dt=now,
        to_dt=now + timedelta(days=days_ahead),
        doctor_id=doc_uuid,
        limit=50,
    )

    return {
        "slots": [
            {
                "id": str(s.id),
                "doctor_id": str(s.doctor_id),
                "slot_start": s.slot_start.isoformat(),
                "slot_end": s.slot_end.isoformat(),
                "is_booked": s.is_booked,
            }
            for s in slots
        ],
        "count": len(slots),
    }


class SeedSlotsRequest(BaseModel):
    doctor_id: str
    days_ahead: int = 14
    slot_duration_minutes: int = 30


@router.post("/seed-slots")
async def seed_slots(
    body: SeedSlotsRequest,
    db: AsyncSession = Depends(db_session),
    _: dict = Depends(require_admin),
):
    """Seed availability slots for a doctor. Requires admin auth."""
    svc = BookingService(db)
    count = await svc.seed_slots(
        doctor_id=uuid.UUID(body.doctor_id),
        days_ahead=body.days_ahead,
        slot_duration_minutes=body.slot_duration_minutes,
    )
    return {"message": f"Seeded {count} slots for doctor {body.doctor_id}", "count": count}


@router.get("/{appointment_id}")
async def get_appointment(
    appointment_id: str,
    db: AsyncSession = Depends(db_session),
):
    """Get full appointment details."""
    result = await db.execute(
        select(Appointment).where(Appointment.id == uuid.UUID(appointment_id))
    )
    appt = result.scalar_one_or_none()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    return {
        "id": str(appt.id),
        "patient_id": str(appt.patient_id) if appt.patient_id else None,
        "doctor_id": str(appt.doctor_id) if appt.doctor_id else None,
        "urgency_level": appt.urgency_level,
        "urgency_score": appt.urgency_score,
        "chief_complaint": appt.chief_complaint,
        "status": appt.status,
        "created_at": appt.created_at.isoformat(),
    }


@router.post("/{appointment_id}/cancel")
async def cancel_appointment(
    appointment_id: str,
    db: AsyncSession = Depends(db_session),
    _: dict = Depends(require_admin),
):
    """Cancel an appointment and free the slot."""
    svc = BookingService(db)
    appt = await svc.cancel_appointment(uuid.UUID(appointment_id))
    return {"status": "cancelled", "appointment_id": str(appt.id)}
