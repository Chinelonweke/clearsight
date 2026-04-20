from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from app.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class TTSService:
    """
    Async TTS using local Piper.
    Synthesis runs in a thread executor to avoid blocking the event loop.
    All errors are non-fatal — returns empty bytes on any failure.
    """

    def __init__(self) -> None:
        self._model_path = Path(settings.piper_model_path)
        # Config path is optional — Piper can infer it from the model path
        self._config_path = Path(str(self._model_path) + ".json")
        self._voice = None
        self._available = False
        self._try_load()

    def _try_load(self) -> None:
        try:
            from piper.voice import PiperVoice  # type: ignore

            if not self._model_path.exists():
                logger.warning(
                    f"Piper model not found at {self._model_path}. TTS unavailable."
                )
                return

            config_path = str(self._config_path) if self._config_path.exists() else None

            self._voice = PiperVoice.load(
                str(self._model_path),
                config_path=config_path,
                use_cuda=False,
            )
            self._available = True
            logger.success(f"Piper TTS loaded | model={self._model_path.name}")

        except ImportError:
            logger.warning(
                "piper-tts not installed. TTS unavailable. "
                "Install with: pip install piper-tts"
            )
        except Exception as exc:
            logger.warning(f"Piper TTS load failed (non-fatal): {exc}")

    def _synthesize_sync(self, text: str) -> bytes:
        """
        Synthesise text to WAV bytes synchronously.
        Uses a temp file to avoid the 'channels not specified' WAV header error
        that occurs when writing directly to a BytesIO buffer with some Piper versions.
        """
        if not self._available or self._voice is None:
            return b""

        tmp_path = None
        try:
            # Write to a real temp WAV file — avoids BytesIO channel config issues
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, dir=tempfile.gettempdir()
            ) as tmp:
                tmp_path = tmp.name

            # Piper writes a proper WAV file with correct headers
            self._voice.synthesize_to_file(text, tmp_path)

            # Read back the complete WAV bytes
            with open(tmp_path, "rb") as f:
                return f.read()

        except AttributeError:
            # Older Piper versions use synthesize(text, wav_file) not synthesize_to_file
            return self._synthesize_sync_legacy(text)

        except Exception as exc:
            logger.warning(f"TTS synthesis failed (non-fatal): {exc}")
            return b""

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _synthesize_sync_legacy(self, text: str) -> bytes:
        """Synthesize using new AudioChunk iterable API."""
        import io, wave, struct
        if not self._voice:
            return b""
        try:
            chunks = list(self._voice.synthesize(text))
            if not chunks:
                return b""
            sample_rate = chunks[0].sample_rate
            all_samples = []
            for chunk in chunks:
                audio = chunk.audio_float_array
                samples = (audio * 32767).astype('int16')
                all_samples.extend(samples.tolist())
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(struct.pack('<' + 'h' * len(all_samples), *all_samples))
            return buf.getvalue()
        except Exception as exc:
            logger.warning(f"TTS legacy synthesis failed (non-fatal): {exc}")
            return b""

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesise text to WAV bytes asynchronously.
        Always returns bytes — empty bytes if TTS unavailable or on any error.
        Never raises.
        """
        if not text or not text.strip():
            return b""

        if not self._available:
            return b""

        # Truncate very long texts to avoid slow synthesis
        if len(text) > 800:
            text = text[:800] + "..."

        logger.debug(
            f"Synthesising TTS | length={len(text)} chars | preview={text[:50]!r}"
        )

        try:
            loop = asyncio.get_running_loop()
            audio_bytes = await loop.run_in_executor(
                None, self._synthesize_sync, text
            )
            if audio_bytes:
                logger.debug(f"TTS complete | size={len(audio_bytes) / 1024:.1f}KB")
            return audio_bytes

        except Exception as exc:
            logger.warning(f"TTS synthesis failed (non-fatal): {exc}")
            return b""

    @property
    def is_available(self) -> bool:
        return self._available