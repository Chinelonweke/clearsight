from __future__ import annotations
"""
app/core/middleware.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Custom ASGI middleware for:
  1. RequestTimingMiddleware  â€” measures every request's wall-clock duration,
     logs it, and attaches X-Process-Time to the response header.
     Emits a WARNING if the request exceeds SLOW_REQUEST_THRESHOLD_MS.

  2. ErrorHandlingMiddleware  â€” catches unhandled exceptions, logs them with
     full tracebacks, and returns a clean JSON error response instead of
     letting FastAPI expose a raw 500.

Mount order in main.py (outermost first):
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestTimingMiddleware)
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.core.exceptions import ClearSightError
from app.core.logger import get_logger

logger = get_logger(__name__)


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """
    Attaches a unique request-id, measures processing time, and logs
    every completed HTTP request with method, path, status, and duration.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]          # short 8-char id for log correlation
        request.state.request_id = request_id

        start = time.perf_counter()

        # Attach request_id to the request so route handlers can read it
        response: Response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"

        log_msg = (
            f"[{request_id}] {request.method} {request.url.path} "
            f"â†’ {response.status_code} | {duration_ms:.1f}ms"
        )

        if not settings.enable_request_logging:
            return response

        if duration_ms > settings.slow_request_threshold_ms:
            logger.warning(f"SLOW REQUEST {log_msg}")
        elif response.status_code >= 500:
            logger.error(log_msg)
        elif response.status_code >= 400:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global exception handler.
    - Known ClearSightError subclasses   â†’ structured JSON with the correct status code
    - Unexpected exceptions              â†’ 500 JSON + full traceback in server logs
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)

        except ClearSightError as exc:
            logger.warning(
                f"Handled error [{exc.status_code}]: {exc.message} "
                f"| path={request.url.path}"
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.__class__.__name__,
                    "message": exc.message,
                    "detail": exc.detail,
                },
            )

        except Exception as exc:
            logger.exception(
                f"Unhandled exception on {request.method} {request.url.path}: {exc}"
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "message": "An unexpected error occurred. Please try again later.",
                    "detail": None,
                },
            )
