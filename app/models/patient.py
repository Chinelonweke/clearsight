"""
app/models/patient.py
─────────────────────────────────────────────────────────
Patient ORM model.
"""

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import Boolean, Date, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    full_name: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(30), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255), index=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date)

    # Flexible storage: allergies, prior diagnoses, preferred language, etc.
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    appointments: Mapped[list["Appointment"]] = relationship(
        "Appointment", back_populates="patient", lazy="selectin"
    )
    intake_forms: Mapped[list["IntakeForm"]] = relationship(
        "IntakeForm", back_populates="patient", lazy="selectin"
    )
    sessions: Mapped[list["ConversationSession"]] = relationship(
        "ConversationSession", back_populates="patient", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Patient id={self.id} name={self.full_name!r} phone={self.phone!r}>"