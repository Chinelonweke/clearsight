"""
app/utils/audio.py
─────────────────────────────────────────────────────────
Audio format detection and conversion utilities.
Used by the STT service and WebSocket voice handler.

All heavy processing is done via pydub (which wraps ffmpeg).
If ffmpeg is not installed, conversion falls back to returning
the raw bytes and logging a warning.
"""

import io
from pathlib import Path

from app.core.logger import get_logger

logger = get_logger(__name__)

# Map browser-sent content-type headers to file extensions
MIME_TO_EXT = {
    "audio/webm": "webm",
    "audio/webm;codecs=opus": "webm",
    "audio/ogg": "ogg",
    "audio/ogg;codecs=opus": "ogg",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/m4a": "m4a",
    "video/webm": "webm",
}


def detect_format_from_bytes(audio_bytes: bytes) -> str:
    """
    Sniff the audio format from the file magic bytes.
    Returns a best-guess extension string.
    """
    if audio_bytes[:4] == b"RIFF":
        return "wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return "mp3"
    if audio_bytes[:4] == b"OggS":
        return "ogg"
    if audio_bytes[:4] in (b"ftyp", b"\x00\x00\x00\x1c"):
        return "mp4"
    # Default — most browsers send webm from MediaRecorder
    return "webm"


def get_filename_for_groq(audio_bytes: bytes, content_type: str | None = None) -> str:
    """
    Return a filename with the correct extension for the Groq Whisper API.
    Groq uses the filename extension to determine the audio codec.
    """
    if content_type:
        # Clean content-type (strip params like ;codecs=opus)
        base_mime = content_type.split(";")[0].strip().lower()
        ext = MIME_TO_EXT.get(base_mime)
        if ext:
            return f"audio.{ext}"

    ext = detect_format_from_bytes(audio_bytes)
    return f"audio.{ext}"


def convert_to_wav(audio_bytes: bytes, source_format: str = "webm") -> bytes:
    """
    Convert audio bytes to WAV format using pydub.
    WAV is universally supported and avoids codec detection issues.

    Returns original bytes unchanged if conversion fails (e.g. ffmpeg missing).
    """
    try:
        from pydub import AudioSegment  # type: ignore

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=source_format)
        wav_buf = io.BytesIO()
        audio.export(wav_buf, format="wav")
        result = wav_buf.getvalue()
        logger.debug(
            f"Audio converted to WAV | "
            f"in={len(audio_bytes)/1024:.1f}KB → out={len(result)/1024:.1f}KB"
        )
        return result

    except ImportError:
        logger.warning("pydub not available — returning raw audio bytes unchanged")
        return audio_bytes
    except Exception as exc:
        logger.warning(f"Audio conversion failed ({exc}) — returning raw bytes")
        return audio_bytes