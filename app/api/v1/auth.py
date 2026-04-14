from __future__ import annotations
"""
app/api/v1/auth.py
─────────────────────────────────────────────────────────
Authentication routes.

POST /api/v1/auth/login         — admin login (username/password → JWT)
POST /api/v1/auth/doctor-login  — doctor login (username/password → JWT)
POST /api/v1/auth/refresh       — exchange refresh token for new access token
"""

import bcrypt as _bcrypt

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.core.exceptions import AuthenticationError
from app.core.logger import get_logger
from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
)
from app.dependencies import db_session
from app.services.analytics_service import track_event

logger = get_logger(__name__)
router = APIRouter()


# ── Request / Response models ──────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# ── Password helper (uses bcrypt directly — avoids passlib version conflicts) ──

def _check_password(plain: str, hashed: str) -> bool:
    """Verify a plain password against a bcrypt hash."""
    try:
        return _bcrypt.checkpw(
            plain.encode("utf-8"),
            hashed.encode("utf-8"),
        )
    except Exception as exc:
        logger.warning(f"Password check error: {exc}")
        return False


# ── Admin login ────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """
    Admin login — compares against ADMIN_USERNAME / ADMIN_PASSWORD in .env.
    Returns JWT access + refresh tokens on success.
    """
    if (
        body.username != settings.admin_username
        or body.password != settings.admin_password
    ):
        logger.warning(f"Failed admin login | username={body.username!r}")
        await track_event("auth_login", success=False, metadata={"username": body.username})
        raise AuthenticationError(message="Invalid username or password.")

    access = create_access_token(
        subject=body.username,
        extra={"role": "admin"}
    )
    refresh = create_refresh_token(subject=body.username)

    logger.info(f"Admin login | username={body.username!r}")
    await track_event("auth_login", success=True, metadata={"username": body.username})

    return TokenResponse(access_token=access, refresh_token=refresh)


# ── Doctor login ───────────────────────────────────────────────────────────────

@router.post("/doctor-login")
async def doctor_login(
    credentials: LoginRequest,
    db: AsyncSession = Depends(db_session),
):
    """
    Doctor login — looks up doctor by username in NeonDB and verifies
    bcrypt password hash. Returns JWT with role=doctor and doctor_id.
    """
    from app.models.doctor import Doctor

    result = await db.execute(
        select(Doctor).where(Doctor.username == credentials.username)
    )
    doctor = result.scalar_one_or_none()

    if not doctor:
        logger.warning(f"Doctor login failed — unknown username: {credentials.username!r}")
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    if not doctor.password_hash:
        logger.warning(f"Doctor {credentials.username!r} has no password set.")
        raise HTTPException(status_code=401, detail="Account not configured. Contact admin.")

    if not _check_password(credentials.password, doctor.password_hash):
        logger.warning(f"Doctor login failed — wrong password | username={credentials.username!r}")
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_access_token(
        subject=doctor.username,
        extra={
            "role": "doctor",
            "doctor_id": str(doctor.id),
            "doctor_name": doctor.full_name,
        }
    )

    logger.info(f"Doctor login | username={doctor.username} | id={doctor.id}")
    await track_event("auth_login", success=True, metadata={"username": doctor.username, "role": "doctor"})

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": "doctor",
        "doctor_id": str(doctor.id),
        "doctor_name": doctor.full_name,
    }


# ── Token refresh ──────────────────────────────────────────────────────────────

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshTokenRequest):
    """Exchange a valid refresh token for a new access token."""
    payload = verify_token(body.refresh_token, expected_type="refresh")
    subject = payload["sub"]

    access = create_access_token(subject=subject, extra={"role": "admin"})
    new_refresh = create_refresh_token(subject=subject)

    logger.info(f"Token refreshed | sub={subject}")
    return TokenResponse(access_token=access, refresh_token=new_refresh)