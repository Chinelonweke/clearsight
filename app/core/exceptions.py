"""
app/core/exceptions.py
─────────────────────────────────────────────────────────
Centralised exception hierarchy for ClearSight.
All custom exceptions inherit from ClearSightError so you
can catch them at the top level if needed. 

Usage:
from app.core.exceptions import PatientNotFoundError
raise PatientNotFoundError(patient_id="abc-123")
"""

from typing import Any


# ── Base ──────────────────────────────────────────────────────────────────────

class ClearSightError(Exception):
    """Base class for all ClearSight application errors."""

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        status_code: int = 500,
        detail: Any = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, status_code={self.status_code})"


# ── Auth ─────────────────────────────────────────────────────────────────────

class AuthenticationError(ClearSightError):
    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message=message, status_code=401)


class AuthorizationError(ClearSightError):
    def __init__(self, message: str = "You do not have permission to perform this action") -> None:
        super().__init__(message=message, status_code=403)


class InvalidTokenError(ClearSightError):
    def __init__(self, message: str = "Token is invalid or expired") -> None:
        super().__init__(message=message, status_code=401)


# ── Not Found ─────────────────────────────────────────────────────────────────

class NotFoundError(ClearSightError):
    def __init__(self, resource: str = "Resource", identifier: Any = None) -> None:
        detail = f"{resource} with id={identifier!r} not found" if identifier else f"{resource} not found"
        super().__init__(message=detail, status_code=404, detail=identifier)


class PatientNotFoundError(NotFoundError):
    def __init__(self, patient_id: Any = None) -> None:
        super().__init__(resource="Patient", identifier=patient_id)


class SessionNotFoundError(NotFoundError):
    def __init__(self, session_id: Any = None) -> None:
        super().__init__(resource="Conversation session", identifier=session_id)


class BookingNotFoundError(NotFoundError):
    def __init__(self, booking_id: Any = None) -> None:
        super().__init__(resource="Appointment booking", identifier=booking_id)


class DoctorNotFoundError(NotFoundError):
    def __init__(self, doctor_id: Any = None) -> None:
        super().__init__(resource="Doctor", identifier=doctor_id)


class SlotNotAvailableError(ClearSightError):
    def __init__(self, slot_id: Any = None) -> None:
        super().__init__(
            message=f"Appointment slot {slot_id!r} is no longer available",
            status_code=409,
            detail=slot_id,
        )


# ── Validation ────────────────────────────────────────────────────────────────

class ValidationError(ClearSightError):
    def __init__(self, message: str = "Validation failed", detail: Any = None) -> None:
        super().__init__(message=message, status_code=422, detail=detail)


class InvalidAudioFormatError(ValidationError):
    def __init__(self) -> None:
        super().__init__(message="Audio format not supported. Use WAV, MP3, WEBM, or OGG.")


class InvalidImageFormatError(ValidationError):
    def __init__(self) -> None:
        super().__init__(message="Image format not supported. Use JPEG, PNG, or WEBP.")


class FileTooLargeError(ValidationError):
    def __init__(self, max_mb: int = 10) -> None:
        super().__init__(message=f"File exceeds maximum size of {max_mb}MB.")


# ── AI / External Services ────────────────────────────────────────────────────

class AIServiceError(ClearSightError):
    def __init__(self, service: str = "AI", message: str = "AI service call failed") -> None:
        super().__init__(
            message=f"{service} error: {message}",
            status_code=502,
        )


class LLMError(AIServiceError):
    def __init__(self, message: str = "LLM inference failed") -> None:
        super().__init__(service="LLM (Groq)", message=message)


class STTError(AIServiceError):
    def __init__(self, message: str = "Speech transcription failed") -> None:
        super().__init__(service="STT (Whisper)", message=message)


class TTSError(AIServiceError):
    def __init__(self, message: str = "Speech synthesis failed") -> None:
        super().__init__(service="TTS (Piper)", message=message)


class VisionError(AIServiceError):
    def __init__(self, message: str = "Image analysis failed") -> None:
        super().__init__(service="Vision (LLaVA)", message=message)


class RAGError(ClearSightError):
    def __init__(self, message: str = "Knowledge retrieval failed") -> None:
        super().__init__(message=f"RAG error: {message}", status_code=500)


# ── Database ──────────────────────────────────────────────────────────────────

class DatabaseError(ClearSightError):
    def __init__(self, message: str = "Database operation failed") -> None:
        super().__init__(message=message, status_code=500)


class DuplicateRecordError(ClearSightError):
    def __init__(self, resource: str = "Record") -> None:
        super().__init__(
            message=f"{resource} already exists",
            status_code=409,
        )


# ── Rate Limiting ──────────────────────────────────────────────────────────────

class RateLimitError(ClearSightError):
    def __init__(self) -> None:
        super().__init__(
            message="Too many requests. Please wait before trying again.",
            status_code=429,
        )