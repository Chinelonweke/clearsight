import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    patient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="SET NULL"), index=True
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    outcome: Mapped[str | None] = mapped_column(String(30))

    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, default=dict
    )

    patient: Mapped["Patient"] = relationship(
        "Patient", back_populates="sessions"
    )

    def __repr__(self) -> str:
        return f"<ConversationSession id={self.id} outcome={self.outcome!r}>"