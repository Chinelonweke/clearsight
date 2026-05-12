"""
app/config.py
─────────────────────────────────────────────────────────
Central settings loaded from environment variables / .env file.
Access anywhere via:  from app.config import settings
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ─────────────────────────────────────────────────
    app_env: str = "development"
    app_name: str = "ClearSight Eye Clinic"
    app_version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"

    # ── Security ────────────────────────────────────────────
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    admin_username: str = "admin"
    admin_password: str

    # ── Groq ────────────────────────────────────────────────
    groq_api_key: str
    groq_llm_model: str = "llama3-70b-8192"
    groq_vision_model: str = "llava-v1.5-7b"
    groq_whisper_model: str = "whisper-large-v3"
    groq_max_tokens: int = 1024
    groq_temperature: float = 0.7

    # ── NeonDB ──────────────────────────────────────────────
    database_url: str

    # ── Redis ───────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_password: str = ""

    # ── ChromaDB ────────────────────────────────────────────
    chroma_host: str = "clearsight_chroma"
    chroma_port: int = 8001
    chroma_collection_name: str = "eye_conditions"

    # ── File Storage ────────────────────────────────────────
    storage_backend: str = "local"
    local_storage_path: str = "data/uploads"

    # ── TTS ─────────────────────────────────────────────────
    piper_model_path: str = "data/tts_models/en_US-lessac-medium.onnx"
    piper_model_config: str = "data/tts_models/en_US-lessac-medium.onnx.json"

    # ── Clinic ──────────────────────────────────────────────
    clinic_name: str = "ClearSight Eye Clinic"
    clinic_phone: str = "+2348000000000"
    clinic_email: str = "info@clearsight.ng"
    clinic_address: str = "Lagos, Nigeria"
    clinic_opening_hour: int = 8
    clinic_closing_hour: int = 18
    clinic_timezone: str = "Africa/Lagos"

    # ── Email ───────────────────────────────────────────────
    resend_api_key: str = ""
    resend_from_email: str = "nwekechinelo25@yahoo.com"
    gmail_user: str = ""
    gmail_app_password: str = ""

    # ── Observability ───────────────────────────────────────
    metrics_db_path: str = "data/metrics.db"
    enable_request_logging: bool = True
    slow_request_threshold_ms: int = 2000

    # ── Session ──────────────────────────────────────────────
    session_ttl_seconds: int = 21600  # 6 hours
    max_history_messages: int = 100  # cap history to avoid unbounded memory

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance — reads .env once and reuses.
    Use @lru_cache so the file is not re-read on every import.
    """
    return Settings()


# Convenience singleton — import this directly
settings: Settings = get_settings()