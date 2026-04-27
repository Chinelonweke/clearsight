from __future__ import annotations
"""
app/api/v1/auth.py
─────────────────────────────────────────────────────────
Authentication routes.

POST /api/v1/auth/login              — admin login (username/password → JWT)
POST /api/v1/auth/doctor-login       — doctor login (username/password → JWT)
POST /api/v1/auth/refresh            — exchange refresh token for new access token
POST /api/v1/auth/register           — patient registration
POST /api/v1/auth/patient-login      — patient login (email/password → JWT)
POST /api/v1/auth/forgot-password    — send password reset email
POST /api/v1/auth/reset-password     — reset password with token
GET  /api/v1/auth/google             — Google OAuth redirect
GET  /api/v1/auth/google/callback    — Google OAuth callback
GET  /api/v1/auth/me                 — get current patient profile
"""

import bcrypt as _bcrypt
import os
import secrets
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, HTMLResponse
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

# ── Google OAuth config ────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI",
    "https://clearsightclinic.online/api/v1/auth/google/callback"
)


# ── Request / Response models ──────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class PatientRegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str


class PatientLoginRequest(BaseModel):
    email: str
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# ── Password helpers ───────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    return _bcrypt.hashpw(
        plain.encode("utf-8"),
        _bcrypt.gensalt(rounds=12)
    ).decode("utf-8")


def _check_password(plain: str, hashed: str) -> bool:
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
    if (
        body.username != settings.admin_username
        or body.password != settings.admin_password
    ):
        logger.warning(f"Failed admin login | username={body.username!r}")
        await track_event("auth_login", success=False, metadata={"username": body.username})
        raise AuthenticationError(message="Invalid username or password.")

    access = create_access_token(subject=body.username, extra={"role": "admin"})
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
    from app.models.doctor import Doctor

    result = await db.execute(
        select(Doctor).where(Doctor.username == credentials.username)
    )
    doctor = result.scalar_one_or_none()

    if not doctor or not doctor.password_hash:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    if not _check_password(credentials.password, doctor.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_access_token(
        subject=doctor.username,
        extra={
            "role": "doctor",
            "doctor_id": str(doctor.id),
            "doctor_name": doctor.full_name,
        }
    )
    logger.info(f"Doctor login | username={doctor.username}")
    await track_event("auth_login", success=True, metadata={"role": "doctor"})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": "doctor",
        "doctor_id": str(doctor.id),
        "doctor_name": doctor.full_name,
    }


# ── Patient registration ───────────────────────────────────────────────────────

@router.post("/register")
async def register_patient(
    body: PatientRegisterRequest,
    db: AsyncSession = Depends(db_session),
):
    from app.models.patient import Patient

    result = await db.execute(
        select(Patient).where(Patient.email == body.email.lower().strip())
    )
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")

    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    patient = Patient(
        full_name=body.full_name.strip(),
        email=body.email.lower().strip(),
        password_hash=_hash_password(body.password),
    )
    db.add(patient)
    await db.commit()
    await db.refresh(patient)

    access = create_access_token(
        subject=str(patient.id),
        extra={"role": "patient", "email": patient.email, "name": patient.full_name}
    )
    refresh = create_refresh_token(subject=str(patient.id))

    logger.info(f"Patient registered | email={patient.email} | id={patient.id}")
    await track_event("patient_registered", metadata={"patient_id": str(patient.id)})

    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "patient_id": str(patient.id),
        "full_name": patient.full_name,
        "email": patient.email,
    }


# ── Patient login ──────────────────────────────────────────────────────────────

@router.post("/patient-login")
async def patient_login(
    body: PatientLoginRequest,
    db: AsyncSession = Depends(db_session),
):
    from app.models.patient import Patient

    result = await db.execute(
        select(Patient).where(Patient.email == body.email.lower().strip())
    )
    patient = result.scalar_one_or_none()

    if not patient or not patient.password_hash:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not _check_password(body.password, patient.password_hash):
        logger.warning(f"Patient login failed | email={body.email}")
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    access = create_access_token(
        subject=str(patient.id),
        extra={"role": "patient", "email": patient.email, "name": patient.full_name}
    )
    refresh = create_refresh_token(subject=str(patient.id))

    logger.info(f"Patient login | email={patient.email}")
    await track_event("patient_login", metadata={"patient_id": str(patient.id)})

    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "patient_id": str(patient.id),
        "full_name": patient.full_name,
        "email": patient.email,
    }


# ── Forgot password ────────────────────────────────────────────────────────────

@router.post("/forgot-password")
async def forgot_password(
    body: ForgotPasswordRequest,
    db: AsyncSession = Depends(db_session),
):
    from app.models.patient import Patient
    from app.models.password_reset import PasswordResetToken

    result = await db.execute(
        select(Patient).where(Patient.email == body.email.lower().strip())
    )
    patient = result.scalar_one_or_none()

    if patient:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        existing = await db.execute(
            select(PasswordResetToken).where(
                PasswordResetToken.patient_id == patient.id,
                PasswordResetToken.used == False,
            )
        )
        for old_token in existing.scalars().all():
            old_token.used = True

        reset_token = PasswordResetToken(
            patient_id=patient.id,
            token=token,
            expires_at=expires_at,
        )
        db.add(reset_token)
        await db.commit()

        reset_url = f"https://clearsightclinic.online/reset-password?token={token}"
        try:
            from app.services.email_service import _send_email
            await _send_email(
                to_email=patient.email,
                subject="Reset your ClearSight password",
                body=f"""Hello {patient.full_name},

You requested a password reset for your ClearSight account.

Click the link below to reset your password (expires in 1 hour):
{reset_url}

If you didn't request this, you can safely ignore this email.

— ClearSight Eye Clinic
"""
            )
        except Exception as exc:
            logger.warning(f"Failed to send reset email: {exc}")

        logger.info(f"Password reset requested | email={body.email}")

    return {"message": "If that email is registered, a reset link has been sent."}


# ── Reset password ─────────────────────────────────────────────────────────────

@router.post("/reset-password")
async def reset_password(
    body: ResetPasswordRequest,
    db: AsyncSession = Depends(db_session),
):
    from app.models.patient import Patient
    from app.models.password_reset import PasswordResetToken

    result = await db.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.token == body.token,
            PasswordResetToken.used == False,
        )
    )
    reset_token = result.scalar_one_or_none()

    if not reset_token:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token.")

    if reset_token.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Reset token has expired. Please request a new one.")

    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    result_p = await db.execute(
        select(Patient).where(Patient.id == reset_token.patient_id)
    )
    patient = result_p.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=400, detail="Patient not found.")

    patient.password_hash = _hash_password(body.new_password)
    reset_token.used = True
    await db.commit()

    logger.info(f"Password reset successful | patient_id={patient.id}")
    return {"message": "Password reset successfully. You can now sign in."}


# ── Google OAuth ───────────────────────────────────────────────────────────────

@router.get("/google")
async def google_oauth_redirect():
    """Redirect to Google OAuth consent screen."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=501, detail="Google OAuth not configured.")
    import urllib.parse
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    return RedirectResponse(url=url, status_code=302)


@router.get("/google/callback")
async def google_oauth_callback(
    code: str | None = None,
    error: str | None = None,
    db: AsyncSession = Depends(db_session),
):
    """Handle Google OAuth callback — create or login patient."""
    if error or not code:
        return RedirectResponse(url="/landing?error=google_cancelled", status_code=302)

    try:
        import httpx
        from app.models.patient import Patient

        async with httpx.AsyncClient(timeout=15.0) as client:
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
            )
            tokens = token_resp.json()
            logger.info(
                f"Google token exchange | status={token_resp.status_code} "
                f"| keys={list(tokens.keys())}"
            )

            if token_resp.status_code != 200 or "access_token" not in tokens:
                logger.error(
                    f"Google token exchange failed | status={token_resp.status_code} "
                    f"| error={tokens.get('error')} | desc={tokens.get('error_description')}"
                )
                return RedirectResponse(url="/landing?error=google_failed", status_code=302)

            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
            userinfo = userinfo_resp.json()
            logger.info(f"Google userinfo | email={userinfo.get('email')}")

        google_email = userinfo.get("email", "").lower()
        google_name = userinfo.get("name", "")

        if not google_email:
            return RedirectResponse(url="/landing?error=google_no_email", status_code=302)

        result = await db.execute(
            select(Patient).where(Patient.email == google_email)
        )
        patient = result.scalar_one_or_none()

        if not patient:
            patient = Patient(
                full_name=google_name,
                email=google_email,
                password_hash=None,
            )
            db.add(patient)
            await db.commit()
            await db.refresh(patient)
            logger.info(f"New patient via Google | email={google_email}")
        else:
            logger.info(f"Existing patient via Google | email={google_email}")

        access = create_access_token(
            subject=str(patient.id),
            extra={"role": "patient", "email": patient.email, "name": patient.full_name}
        )

        return RedirectResponse(url=f"/app?token={access}", status_code=302)

    except Exception as exc:
        logger.error(f"Google OAuth callback error: {exc}")
        return RedirectResponse(url="/landing?error=google_failed", status_code=302)


# ── Get current patient ────────────────────────────────────────────────────────

@router.get("/me")
async def get_current_patient(
    request: Request,
    db: AsyncSession = Depends(db_session),
):
    from app.models.patient import Patient

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated.")

    token = auth_header.split(" ")[1]
    try:
        payload = verify_token(token, expected_type="access")
        patient_id = payload["sub"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    result = await db.execute(
        select(Patient).where(Patient.id == patient_id)
    )
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    return {
        "patient_id": str(patient.id),
        "full_name": patient.full_name,
        "email": patient.email,
        "created_at": patient.created_at.isoformat(),
    }


# ── Token refresh ──────────────────────────────────────────────────────────────

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshTokenRequest):
    payload = verify_token(body.refresh_token, expected_type="refresh")
    subject = payload["sub"]
    access = create_access_token(subject=subject, extra={"role": "admin"})
    new_refresh = create_refresh_token(subject=subject)
    logger.info(f"Token refreshed | sub={subject}")
    return TokenResponse(access_token=access, refresh_token=new_refresh)