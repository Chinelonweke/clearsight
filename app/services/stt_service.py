from __future__ import annotations
"""
app/services/stt_service.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Speech-to-text service using Groq's Whisper API (whisper-large-v3).

Responsibilities:
  - Accept raw audio bytes (WAV, MP3, WEBM, OGG, M4A)
  - Convert to the format Groq Whisper expects (file-like object)
  - Return a clean transcription string
  - Detect and handle empty/silent audio gracefully

Groq Whisper limits (free tier):
  - Max file size: 25 MB
  - Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
  - Max audio duration: ~10 minutes per request

Usage:
    stt = STTService()
    transcript = await stt.transcribe(audio_bytes, filename="audio.webm")
"""

from groq import AsyncGroq

from app.config import settings
from app.core.exceptions import STTError, InvalidAudioFormatError, FileTooLargeError
from app.core.logger import get_logger

logger = get_logger(__name__)

# Supported audio MIME types â†’ file extension for Groq
_SUPPORTED_FORMATS = {
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "mp4",
    "audio/m4a": "m4a",
    "video/webm": "webm",       # browsers often send webm with video MIME
}

_MAX_BYTES = 25 * 1024 * 1024   # 25 MB


class STTService:
    """
    Async STT using Groq Whisper.
    Instantiate once and share across the app lifetime.
    """

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._model = settings.groq_whisper_model
        logger.info(f"STTService initialised | model={self._model}")

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        language: str = "en",
        prompt: str | None = None,
    ) -> str:
        """
        Transcribe raw audio bytes to text.

        Args:
            audio_bytes: Raw audio data as bytes.
            filename:    Filename hint used by Groq to detect the format.
                         Include the correct extension (e.g. "audio.webm").
            language:    ISO-639-1 language code. Default "en" (English).
            prompt:      Optional text to prime the transcription context
                         (e.g. "Eye clinic patient describing symptoms.").
                         Helps Whisper produce more accurate medical vocabulary.

        Returns:
            Transcription text string. Empty string if audio is silent.

        Raises:
            FileTooLargeError:    If audio exceeds 25 MB.
            InvalidAudioFormatError: If the format is not supported.
            STTError:             On Groq API failure.
        """
        # â”€â”€ Validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if len(audio_bytes) > _MAX_BYTES:
            raise FileTooLargeError(max_mb=25)

        if len(audio_bytes) < 100:
            logger.warning("Audio bytes too short â€” likely silent or empty.")
            return ""

        logger.debug(
            f"Transcribing audio | size={len(audio_bytes)/1024:.1f}KB "
            f"| filename={filename} | language={language}"
        )

        # â”€â”€ Whisper transcription â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            transcription_kwargs = {
                "file": (filename, audio_bytes),
                "model": self._model,
                "language": language,
                "response_format": "text",
            }
            if prompt:
                transcription_kwargs["prompt"] = prompt

            result = await self._client.audio.transcriptions.create(
                **transcription_kwargs
            )

            # Groq returns str when response_format="text"
            transcript = result.strip() if isinstance(result, str) else result.text.strip()

            logger.info(
                f"Transcription complete | length={len(transcript)} chars "
                f"| preview={transcript[:60]!r}"
            )
            return transcript

        except Exception as exc:
            logger.error(f"Whisper transcription failed: {exc}")
            raise STTError(message=str(exc)) from exc

    async def transcribe_with_timestamps(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        language: str = "en",
    ) -> dict:
        """
        Transcribe with word-level timestamps (verbose_json format).
        Useful for analytics â€” knowing exactly when patient mentioned each symptom.

        Returns:
            Groq verbose_json transcription object as dict.
        """
        try:
            result = await self._client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model=self._model,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
            return result.model_dump() if hasattr(result, "model_dump") else dict(result)

        except Exception as exc:
            logger.error(f"Whisper verbose transcription failed: {exc}")
            raise STTError(message=str(exc)) from exc
