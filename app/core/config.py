"""
app/core/config.py
──────────────────
Centralized settings loaded from environment variables.
Uses pydantic-settings for type-safe config management.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = "Masar-AI"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # ── MongoDB ────────────────────────────────────────────────────────────────
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "masar_ai"

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    use_openai: bool = False

    # ── GitHub ─────────────────────────────────────────────────────────────────
    github_token: str = ""

    # ── Embedding ──────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── FAISS ──────────────────────────────────────────────────────────────────
    faiss_index_path: str = "./data/faiss_index"
    faiss_freelancer_index: str = "freelancers.index"
    faiss_project_index: str = "projects.index"

    # ── Ranking Weights ────────────────────────────────────────────────────────
    weight_skill_relevance: float = 0.40
    weight_portfolio_quality: float = 0.25
    weight_client_rating: float = 0.20
    weight_response_speed: float = 0.15

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "./logs/masar_ai.log"


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance (singleton)."""
    return Settings()


settings = get_settings()