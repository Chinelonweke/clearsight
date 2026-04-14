from __future__ import annotations
"""
app/models/doctor.py
─────────────────────────────────────────────────────────
Doctor and AvailabilitySlot ORM models.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.booking import Appointment

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Doctor(Base):
    __tablename__ = "doctors"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    specialty: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # ── Auth columns (added for doctor dashboard login) ────────────────────────
    username: Mapped[str | None] = mapped_column(
        String(50), unique=True, nullable=True, index=True
    )
    password_hash: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )

    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    availability_slots: Mapped[list["AvailabilitySlot"]] = relationship(
        "AvailabilitySlot", back_populates="doctor", lazy="selectin"
    )
    appointments: Mapped[list["Appointment"]] = relationship(
        "Appointment", back_populates="doctor", lazy="selectin"
    )

    def __repr__(self) -> str:
        return (
            f"<Doctor id={self.id} name={self.full_name!r} "
            f"username={self.username!r} specialty={self.specialty!r}>"
        )


class AvailabilitySlot(Base):
    __tablename__ = "availability_slots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("doctors.id", ondelete="CASCADE"), index=True
    )
    slot_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    slot_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    is_booked: Mapped[bool] = mapped_column(Boolean, default=False)

    # ── Relationships ──────────────────────────────────────────────────────────
    doctor: Mapped["Doctor"] = relationship(
        "Doctor", back_populates="availability_slots"
    )
    appointment: Mapped["Appointment | None"] = relationship(
        "Appointment", back_populates="slot", uselist=False
    )

    def __repr__(self) -> str:
        return (
            f"<AvailabilitySlot id={self.id} doctor={self.doctor_id} "
            f"start={self.slot_start} booked={self.is_booked}>"
        )