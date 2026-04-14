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

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.logger import get_logger
from app.core.middleware import ErrorHandlingMiddleware, RequestTimingMiddleware
from app.db.neon import dispose_db, init_db
from app.db.redis_client import close_redis, init_redis
from app.rag.chroma_client import init_chroma

logger = get_logger(__name__)


# ── Lifespan: startup + shutdown ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup, then yields (app serves requests),
    then runs the shutdown block.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name}  v{settings.app_version}")
    logger.info(f"  Environment : {settings.app_env.upper()}")
    logger.info("=" * 60)

    # Ensure runtime data directories exist
    for directory in ["logs", "data/uploads", "data/tts_models", "data/knowledge_base"]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    await init_db()
    await init_redis()

    try:
        await init_chroma()
    except Exception as exc:
        logger.warning(f"ChromaDB init warning (non-fatal): {exc}")

    logger.success("All services initialised — ClearSight is ready.")
    logger.info(f"  Docs: http://localhost:8000/docs")
    logger.info(f"  Dashboard: http://localhost:8000/api/v1/admin/dashboard")

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
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
    docs_url="/docs" if not settings.is_production else None,   # hide Swagger in prod
    redoc_url="/redoc" if not settings.is_production else None,
)


# ── Middleware (outermost first) ───────────────────────────────────────────────
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [
        "https://clearsight.ng",
        "https://app.clearsight.ng",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static files (dashboard assets) ───────────────────────────────────────────
static_dir = Path("app/dashboard/static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Routers ────────────────────────────────────────────────────────────────────
# Import here (after middleware) to avoid circular imports at module load time
from app.api.v1.router import v1_router          # noqa: E402
from app.api.websocket import router as ws_router  # noqa: E402

app.include_router(v1_router, prefix="/api/v1")
app.include_router(ws_router)


# ── Root health check ─────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Lightweight health endpoint.
    Used by Docker health checks, load balancers, and deployment platforms.
    """
    logger.debug("Health check called")
    return JSONResponse(
        content={
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_env,
        }
    )


@app.get("/", include_in_schema=False)
async def serve_frontend():
    index_file = Path(__file__).parent / "dashboard" / "templates" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return JSONResponse({"message": f"Welcome to {settings.app_name}", "docs": "/docs"})

@app.get("/staff", include_in_schema=False)
async def serve_staff_dashboard():
    staff_file = Path(__file__).parent / "dashboard" / "templates" / "staff.html"
    if staff_file.exists():
        return FileResponse(str(staff_file), media_type="text/html")
    return JSONResponse({"error": "Staff dashboard not found"})