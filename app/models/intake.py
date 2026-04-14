import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base

if TYPE_CHECKING:
    from app.models.patient import Patient


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class IntakeForm(Base):
    __tablename__ = "intake_forms"
    __table_args__ = (
        CheckConstraint(
            "patient_id IS NOT NULL OR session_id IS NOT NULL",
            name="ck_intake_forms_patient_or_session",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    patient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), index=True
    )

    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), index=True
    )

    symptoms: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    vision_changes: Mapped[str | None] = mapped_column(Text)
    pain_level: Mapped[int | None] = mapped_column(Integer)
    duration: Mapped[str | None] = mapped_column(String(100))
    eye_affected: Mapped[str | None] = mapped_column(String(10))

    image_urls: Mapped[list] = mapped_column(JSONB, default=list)

    raw_transcript: Mapped[str | None] = mapped_column(Text)
    ai_summary: Mapped[str | None] = mapped_column(Text)

    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    patient: Mapped["Patient"] = relationship(
        "Patient", back_populates="intake_forms"
    )

    def __repr__(self) -> str:
        return f"<IntakeForm id={self.id} patient={self.patient_id}>"