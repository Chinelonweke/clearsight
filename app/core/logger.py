"""
app/core/logger.py
─────────────────────────────────────────────────────────
Loguru-based logging with:
  • Green timestamps on all levels
  • Yellow  for WARNING
  • Red     for ERROR / CRITICAL
  • Cyan    for DEBUG module names
  • Rotating file logs (JSON format) for production parsing
  • Async-safe (enqueue=True) 

Usage anywhere in the project:
    from app.core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Service started")
    logger.success("Patient booked successfully")
    logger.warning("Slow response detected")
    logger.error("Database connection failed")
"""

import logging
import sys
from pathlib import Path

from loguru import logger as _loguru_logger

from app.config import settings

# ── Ensure log directory exists ────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ── Remove the default Loguru handler ─────────────────────────────────────────
_loguru_logger.remove()



import re as _re

# ── PII scrubber ───────────────────────────────────────────────────────────────
# Masks patient PII before any log record is written to any sink.
# Healthcare standard: logs must never contain names, emails, or phone numbers.
_PII_PATTERNS = [
    # Email addresses → j***@***.***
    (_re.compile(r'[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}'),
     lambda m: m.group(0)[0] + '***@***.***'),
    # email=value patterns
    (_re.compile(r'(email[=:][\s]*)([^\s|,}\]]+)', _re.IGNORECASE),
     lambda m: m.group(1) + '***@***.***'),
    # name=Full Name patterns
    (_re.compile(r'(name[=:][\s]*)([A-Za-z]+(?:\s+[A-Za-z]+)+)', _re.IGNORECASE),
     lambda m: m.group(1) + m.group(2)[0] + '***'),
    # Nigerian phone numbers
    (_re.compile(r'(\+?234|0)[789][01]\d{8}'),
     lambda m: m.group(0)[:4] + '****' + m.group(0)[-3:]),
    # preview= content (session message previews)
    (_re.compile(r'(preview=["])[^"]{20,}(["])'),
     lambda m: m.group(1) + '[REDACTED]' + m.group(2)),
]


def _scrub_pii(record: dict) -> bool:
    """Loguru filter: mask PII in message before writing to any sink."""
    msg = record["message"]
    for pattern, replacement in _PII_PATTERNS:
        msg = pattern.sub(replacement, msg)
    record["message"] = msg
    return True

def _configure_logger() -> None:
    """
    Configure Loguru with two sinks:
    1. Colourised stdout for development visibility
    2. Rotating JSON file for production analysis and grep-ability
    """

    # ── Sink 1: Colourised Console ─────────────────────────────────────────────
    # Loguru applies colour based on log level automatically when colorize=True.
    # Levels and their default colours:
    #   TRACE    → grey
    #   DEBUG    → blue
    #   INFO     → normal (white/default)
    #   SUCCESS  → bold green       ← custom level built into Loguru
    #   WARNING  → yellow
    #   ERROR    → red
    #   CRITICAL → bold red background
    _loguru_logger.add(
        sys.stdout,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <9}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
            " ─ <level>{message}</level>"
        ),
        level=settings.log_level,
        filter=_scrub_pii,
        enqueue=True,           # async-safe: log calls do not block the event loop
        backtrace=True,         # full stack trace on exceptions
        diagnose=settings.is_development,  # variable values in tracebacks (dev only)
    )

    # ── Sink 2: Rotating JSON file ─────────────────────────────────────────────
    # Written as JSON-lines so you can grep/parse with jq or any log aggregator.
    # Each day gets its own file; files older than 30 days are deleted.
    _loguru_logger.add(
        str(LOG_DIR / "clearsight_{time:YYYY-MM-DD}.log"),
        rotation="00:00",           # rotate at midnight
        retention="30 days",
        compression="gz",           # gzip old files to save disk
        format="{time} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        filter=_scrub_pii,
        enqueue=True,
        serialize=True,             # emit JSON-lines
        backtrace=True,
        diagnose=False,             # no sensitive variable values in prod files
    )


_configure_logger()


def get_logger(name: str):
    """
    Return a Loguru logger bound with the calling module's name.
    The 'name' appears in the cyan module field of every log line.

    Example:
        logger = get_logger(__name__)
        logger.info("Starting triage assessment")
    """
    return _loguru_logger.bind(name=name)


# ── Intercept standard library logging ────────────────────────────────────────
# Third-party libraries (SQLAlchemy, uvicorn, etc.) use stdlib logging.
# This redirects all of it through Loguru so everything appears consistently.
class _InterceptHandler(logging.Handler):
    """Forward stdlib log records into Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Redirect all stdlib loggers through Loguru
logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)

for _lib in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
    logging.getLogger(_lib).handlers = [_InterceptHandler()]