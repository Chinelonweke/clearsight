"""
app/db/neon.py
─────────────────────────────────────────────────────────
NeonDB (PostgreSQL) initialisation helpers.

- init_db()  : Called once at app startup — creates all tables if they don't
               exist (development) and verifies the connection.
- dispose_db(): Called at shutdown to cleanly close pool connections.

In production, table creation is handled by Alembic migrations, not
create_all(). The create_all() call here is a safety net for development
and first-run setup.
"""

from sqlalchemy import text

from app.core.logger import get_logger
from app.models.database import Base, engine

# Import all models so SQLAlchemy's metadata knows about them
from app.models.patient import Patient          # noqa: F401
from app.models.doctor import Doctor, AvailabilitySlot  # noqa: F401
from app.models.booking import Appointment      # noqa: F401
from app.models.intake import IntakeForm                    # noqa: F401
from app.models.session import ConversationSession          # noqa: F401
from app.models.analytics import AnalyticsEvent             # noqa: F401

logger = get_logger(__name__)


async def init_db() -> None:
    """
    Verify the database connection and create all tables if they don't exist.
    Called during FastAPI lifespan startup.
    """
    logger.info("Connecting to NeonDB (PostgreSQL)...")

    try:
        async with engine.begin() as conn:
            # Verify connectivity with a cheap query
            await conn.execute(text("SELECT 1"))
            logger.success("NeonDB connection verified.")

            # Create tables (safe — skips existing tables)
            await conn.run_sync(Base.metadata.create_all)
            logger.success("Database schema synchronised.")

    except Exception as exc:
        logger.error(f"NeonDB initialisation failed: {exc}")
        raise


async def dispose_db() -> None:
    """Dispose the connection pool on shutdown."""
    await engine.dispose()
    logger.info("NeonDB connection pool disposed.")