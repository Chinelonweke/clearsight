"""
app/api/v1/router.py
─────────────────────────────────────────────────────────
Mounts all v1 API route modules onto a single APIRouter.
"""

from fastapi import APIRouter

from app.api.v1 import auth, triage, session, booking, admin

v1_router = APIRouter()

v1_router.include_router(auth.router,    prefix="/auth",    tags=["Auth"])
v1_router.include_router(session.router, prefix="/session", tags=["Sessions"])
v1_router.include_router(triage.router,  prefix="/triage",  tags=["Triage"])
v1_router.include_router(booking.router, prefix="/booking", tags=["Booking"])
v1_router.include_router(admin.router,   prefix="/admin",   tags=["Admin"])