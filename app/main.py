"""
app/main.py
─────────────────────────────────────────────────────────
ClearSight FastAPI application entry point.

Startup sequence (lifespan):
  1. Configure colour logging
  2. Connect to NeonDB and sync schema
  3. Connect to Redis
  4. Initialise ChromaDB collection
  5. Initialise local metrics DB
  6. Mount routers and middleware

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from pathlib import Path

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.logger import get_logger
from app.core.middleware import ErrorHandlingMiddleware, RequestTimingMiddleware
from app.db.neon import dispose_db, init_db
from app.db.redis_client import close_redis, init_redis
from app.rag.chroma_client import init_chroma
from app.services.analytics_service import init_metrics_db, track_event


logger = get_logger(__name__)


# ── Lifespan: startup + shutdown ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name}  v{settings.app_version}")
    logger.info(f"  Environment : {settings.app_env.upper()}")
    logger.info("=" * 60)

    for directory in ["logs", "data/uploads", "data/tts_models", "data/knowledge_base"]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    await init_db()
    await init_redis()
    init_metrics_db()

    try:
        await init_chroma()
    except Exception as exc:
        logger.warning(f"ChromaDB init warning (non-fatal): {exc}")

    logger.success("All services initialised — ClearSight is ready.")
    logger.info(f"  Docs: http://localhost:8000/docs")
    logger.info(f"  Dashboard: http://localhost:8000/api/v1/admin/dashboard")

    yield

    logger.warning("ClearSight shutting down...")
    await dispose_db()
    await close_redis()
    logger.info("All connections closed. Goodbye.")


# ── App instance ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ClearSight — Eye Clinic Triage API",
    description=(
        "AI-powered smart front desk for Nigerian eye clinics.\n\n"
        "Handles voice triage, eye image analysis, intake form auto-fill, "
        "urgency scoring, and appointment booking."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)


# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [
        "https://clearsightclinic.online",
        "https://www.clearsightclinic.online",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static files ───────────────────────────────────────────────────────────────
static_dir = Path("app/dashboard/static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Routers ────────────────────────────────────────────────────────────────────
from app.api.v1.router import v1_router          # noqa: E402
from app.api.websocket import router as ws_router  # noqa: E402

app.include_router(v1_router, prefix="/api/v1")
app.include_router(ws_router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.api_route("/health", methods=["GET", "HEAD"], tags=["Health"])
async def health_check():
    logger.debug("Health check called")
    return JSONResponse(
        content={
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_env,
        }
    )


# ── Frontend routes ───────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_landing():
    """
    Root route — serves landing page.
    Authenticated users are redirected to /app by the landing page JS.
    Unauthenticated users see sign up / login.
    """
    landing_file = Path(__file__).parent / "dashboard" / "templates" / "landing.html"
    if landing_file.exists():
        return FileResponse(str(landing_file), media_type="text/html")
    return JSONResponse({"message": f"Welcome to {settings.app_name}", "docs": "/docs"})


@app.get("/app", include_in_schema=False)
async def serve_app():
    """
    Triage app — serves the main triage interface.
    Only accessible after patient login (enforced client-side via JWT).
    """
    index_file = Path(__file__).parent / "dashboard" / "templates" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return JSONResponse({"error": "App not found"})


@app.get("/landing", include_in_schema=False)
async def serve_landing_explicit():
    """Explicit /landing alias — same as /"""
    landing_file = Path(__file__).parent / "dashboard" / "templates" / "landing.html"
    if landing_file.exists():
        return FileResponse(str(landing_file), media_type="text/html")
    return JSONResponse({"message": f"Welcome to {settings.app_name}"})


@app.get("/staff", include_in_schema=False)
async def serve_staff_dashboard():
    staff_file = Path(__file__).parent / "dashboard" / "templates" / "staff.html"
    if staff_file.exists():
        return FileResponse(str(staff_file), media_type="text/html")
    return JSONResponse({"error": "Staff dashboard not found"})


@app.get("/reset-password", include_in_schema=False)
async def serve_reset_password():
    """Password reset page — reads token from URL query param."""
    landing_file = Path(__file__).parent / "dashboard" / "templates" / "landing.html"
    if landing_file.exists():
        return FileResponse(str(landing_file), media_type="text/html")
    return JSONResponse({"error": "Page not found"})