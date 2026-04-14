
"""
app/models/booking.py
─────────────────────────────────────────────────────────
Appointment (booking) ORM model.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.patient import Patient
    from app.models.doctor import Doctor, AvailabilitySlot

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Appointment(Base):
    __tablename__ = "appointments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="SET NULL"), index=True
    )
    doctor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("doctors.id", ondelete="SET NULL"), index=True
    )
    slot_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("availability_slots.id", ondelete="SET NULL")
    )

    # Triage output
    urgency_level: Mapped[str] = mapped_column(String(20), nullable=False)   # emergency|urgent|routine
    urgency_score: Mapped[int] = mapped_column(Integer, nullable=False)       # 1–10
    chief_complaint: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)

    # Lifecycle status
    status: Mapped[str] = mapped_column(String(20), default="scheduled")      # scheduled|completed|cancelled

    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    patient: Mapped["Patient"] = relationship("Patient", back_populates="appointments")
    doctor: Mapped["Doctor"] = relationship("Doctor", back_populates="appointments")
    slot: Mapped["AvailabilitySlot"] = relationship(
        "AvailabilitySlot", back_populates="appointment"
    )

    def __repr__(self) -> str:
        return (
            f"<Appointment id={self.id} urgency={self.urgency_level} "
            f"status={self.status}>"
        )
