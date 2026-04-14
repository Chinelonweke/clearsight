"""
app/db/migrations/env.py
─────────────────────────────────────────────────────────
Alembic migration environment — configured for async SQLAlchemy + NeonDB.

Run migrations:
    alembic upgrade head          # apply all pending migrations
    alembic revision --autogenerate -m "add patients table"  # generate new migration
    alembic downgrade -1          # roll back one migration
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import settings
from app.models.database import Base

# Import all models so Alembic can detect schema changes
from app.models.patient import Patient          # noqa: F401
from app.models.doctor import Doctor, AvailabilitySlot  # noqa: F401
from app.models.booking import Appointment      # noqa: F401
from app.models.intake import IntakeForm                    # noqa: F401
from app.models.session import ConversationSession          # noqa: F401
from app.models.analytics import AnalyticsEvent             # noqa: F401

# ── Alembic Config ─────────────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override the sqlalchemy.url from alembic.ini with the value from settings
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


# ── Offline migrations (generate SQL without a live DB connection) ─────────────
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online migrations (run against the live async NeonDB connection) ───────────
def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,        # no pool needed for migration runs
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()