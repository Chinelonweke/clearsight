"""
app/utils/image.py
─────────────────────────────────────────────────────────
Image preprocessing and local file storage helpers.

Responsibilities:
  - Validate image bytes (magic bytes check)
  - Resize large images before sending to LLaVA (saves tokens + latency)
  - Save uploaded images to local storage with a timestamped filename
  - Return the storage path for persisting in the database
"""

import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Magic bytes for format detection
_MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG": "png",
    b"RIFF": "webp",     # WEBP starts with RIFF....WEBP
    b"WEBP": "webp",
}

_MAX_DIMENSION = 1024       # pixels — resize if larger
_QUALITY = 85               # JPEG compression quality after resize


def detect_image_mime(image_bytes: bytes) -> str:
    """
    Detect image MIME type from magic bytes.
    Returns e.g. "image/jpeg", "image/png", "image/webp".
    Raises ValueError if format is unrecognised.
    """
    for magic, fmt in _MAGIC.items():
        if image_bytes[:len(magic)] == magic:
            return f"image/{fmt}"
    # Secondary WEBP check
    if len(image_bytes) > 12 and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Unsupported image format. Only JPEG, PNG, and WEBP are accepted.")


def preprocess_image(image_bytes: bytes) -> tuple[bytes, str]:
    """
    Resize and re-encode the image if it exceeds _MAX_DIMENSION on either axis.
    This reduces token usage when sending to LLaVA.

    Returns:
        Tuple of (processed_bytes, mime_type).
    """
    try:
        from PIL import Image  # type: ignore

        mime = detect_image_mime(image_bytes)
        img = Image.open(io.BytesIO(image_bytes))

        # Convert palette or RGBA images to RGB (JPEG does not support RGBA)
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGB")

        # Resize if necessary
        w, h = img.size
        if w > _MAX_DIMENSION or h > _MAX_DIMENSION:
            scale = _MAX_DIMENSION / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
            logger.debug(f"Image resized: {w}x{h} → {new_size[0]}x{new_size[1]}")

        # Re-encode as JPEG for consistency
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=_QUALITY, optimize=True)
        result = out.getvalue()
        logger.debug(
            f"Image preprocessed | "
            f"in={len(image_bytes)/1024:.1f}KB → out={len(result)/1024:.1f}KB"
        )
        return result, "image/jpeg"

    except ImportError:
        logger.warning("Pillow not installed — skipping image preprocessing")
        mime = detect_image_mime(image_bytes)
        return image_bytes, mime
    except Exception as exc:
        logger.warning(f"Image preprocessing failed ({exc}) — using original bytes")
        mime = detect_image_mime(image_bytes)
        return image_bytes, mime


def save_image_locally(
    image_bytes: bytes,
    session_id: str,
    filename: str | None = None,
) -> str:
    """
    Save image bytes to the local upload directory.
    Returns the relative path string for storing in the database.

    File layout:
        data/uploads/images/{session_id}/{timestamp}_{short_hash}.jpg
    """
    upload_dir = Path(settings.local_storage_path) / "images" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.md5(image_bytes).hexdigest()[:8]
    fname = filename or f"{timestamp}_{short_hash}.jpg"
    file_path = upload_dir / fname

    file_path.write_bytes(image_bytes)
    relative_path = str(file_path)

    logger.info(f"Eye image saved | path={relative_path} | size={len(image_bytes)/1024:.1f}KB")
    return relative_path