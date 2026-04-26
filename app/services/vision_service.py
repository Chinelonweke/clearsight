from __future__ import annotations
"""
app/services/vision_service.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Eye image analysis service using Groq's LLaVA vision model.

Responsibilities:
  - Accept uploaded eye images (JPEG, PNG, WEBP)
  - Encode to base64 and send to Groq LLaVA
  - Return structured clinical observations
  - Log every analysis with a session reference

Important disclaimer:
  This service provides AI-assisted *visual observations* to support the
  human triage process â€” NOT a diagnosis. All observations are framed as
  "visual features noted" and must be reviewed by a qualified optometrist.

Usage:
    vision = VisionService()
    result = await vision.analyze_eye_image(image_bytes, session_id="abc")
"""

import base64
import re
from pathlib import Path

from groq import AsyncGroq

from app.config import settings
from app.core.exceptions import VisionError, InvalidImageFormatError, FileTooLargeError
from app.core.logger import get_logger

logger = get_logger(__name__)

# Supported image MIME types and their base64 prefix
_SUPPORTED_MIME = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "image/webp": "image/webp",
}

_MAX_IMAGE_BYTES = 4 * 1024 * 1024    # 4 MB per image (LLaVA limit)

# System prompt that guides LLaVA to respond as a clinical observer, not a diagnostician
_VISION_SYSTEM = """You are an AI visual observer assisting a Nigerian optometrist eye clinic.
Your role is to describe visible features in eye photographs to support clinical triage.

Rules you must always follow:
1. Describe ONLY what you can visually observe â€” do not diagnose.
2. Use clinical but accessible language appropriate for a triage nurse.
3. Note redness, discharge, opacity, swelling, asymmetry, abnormal features.
4. Flag urgency indicators: "This finding may require urgent assessment."
5. Always end with: "These are visual observations only. Clinical examination is required."

Output structure (always use this format):
- Visible features: [what you observe]
- Notable findings: [any clinically significant features]
- Urgency flag: [none / possible / likely]
- Observer note: "These are visual observations only. Clinical examination is required."
"""


class VisionService:
    """
    Async eye image analysis via Groq LLaVA.
    """

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._model = settings.groq_vision_model
        logger.info(f"VisionService initialised | model={self._model}")

    def _encode_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """Base64-encode image bytes for the Groq multimodal message."""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def analyze_eye_image(
        self,
        image_bytes: bytes,
        session_id: str | None = None,
        mime_type: str = "image/jpeg",
        additional_context: str | None = None,
    ) -> dict:
        """
        Analyse an eye photograph and return structured clinical observations.

        Args:
            image_bytes:        Raw image bytes.
            session_id:         Session ID for log correlation.
            mime_type:          MIME type of the image.
            additional_context: Patient-reported symptoms to give the model context.
                                E.g. "Patient reports redness and pain for 2 days."

        Returns:
            dict with keys:
              - raw_observation (str): Full LLaVA text response
              - visible_features (str): Extracted visible features
              - notable_findings (str): Extracted notable findings
              - urgency_flag (str): "none" | "possible" | "likely"
              - disclaimer (str): Always present

        Raises:
            InvalidImageFormatError: If mime_type is not supported.
            FileTooLargeError:       If image exceeds 4 MB.
            VisionError:             On Groq API failure.
        """
        # â”€â”€ Validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if mime_type not in _SUPPORTED_MIME:
            raise InvalidImageFormatError()

        if len(image_bytes) > _MAX_IMAGE_BYTES:
            raise FileTooLargeError(max_mb=4)

        if len(image_bytes) < 500:
            raise VisionError(message="Image file appears to be empty or corrupt.")

        logger.info(
            f"Eye image analysis starting | session={session_id} "
            f"| size={len(image_bytes)/1024:.1f}KB | mime={mime_type}"
        )

        # â”€â”€ Build multimodal message â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        b64_image = self._encode_image(image_bytes, mime_type)
        data_url = f"data:{_SUPPORTED_MIME[mime_type]};base64,{b64_image}"

        user_text = "Please analyse this eye photograph and describe the visible features."
        if additional_context:
            user_text += f"\n\nPatient context: {additional_context}"

        messages = [
            {"role": "system", "content": _VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            },
        ]

        # â”€â”€ Groq LLaVA call â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=600,
                temperature=0.2,    # low temperature for consistent clinical descriptions
            )
            raw_text = response.choices[0].message.content or ""
            logger.info(
                f"Eye image analysis complete | session={session_id} "
                f"| response_length={len(raw_text)}"
            )

        except Exception as exc:
            logger.error(f"LLaVA vision analysis failed | session={session_id}: {exc}")
             # Graceful fallback — don't crash the session
            return {
        "raw_observation": "Vision analysis temporarily unavailable.",
        "visible_features": "Image received but could not be analysed at this time. Please describe your symptoms verbally.",
        "notable_findings": "None — manual analysis unavailable.",
        "urgency_flag": "none",
        "disclaimer": "These are visual observations only. Clinical examination is required.",
    }

        # â”€â”€ Parse structured fields from the response â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        return self._parse_observation(raw_text)

    def _parse_observation(self, raw_text: str) -> dict:
        """
        Extract structured fields from the LLaVA free-text response.
        Uses simple line-by-line parsing â€” robust enough for the structured format
        we enforce in the system prompt.
        """
        lines = raw_text.strip().split("\n")

        visible_features = ""
        notable_findings = ""
        urgency_flag = "none"

        for line in lines:
            line_lower = line.lower()
            if "visible features:" in line_lower:
                visible_features = line.split(":", 1)[-1].strip()
            elif "notable findings:" in line_lower:
                notable_findings = line.split(":", 1)[-1].strip()
            elif "urgency flag:" in line_lower:
                flag_text = line.split(":", 1)[-1].strip().lower()
                if "likely" in flag_text:
                    urgency_flag = "likely"
                elif "possible" in flag_text:
                    urgency_flag = "possible"
                else:
                    urgency_flag = "none"

        return {
            "raw_observation": raw_text,
            "visible_features": visible_features or "Unable to extract â€” see raw observation.",
            "notable_findings": notable_findings or "None noted.",
            "urgency_flag": urgency_flag,
            "disclaimer": "These are visual observations only. Clinical examination is required.",
        }
