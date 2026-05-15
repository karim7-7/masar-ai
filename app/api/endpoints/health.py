"""
app/api/endpoints/health.py
────────────────────────────
GET /health  — Service health check with dependency status
"""

from fastapi import APIRouter
from loguru import logger
from app.schemas.schemas import HealthResponse
from app.core.config import settings

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """
    Returns service health status.
    Checks: MongoDB, FAISS index, embedding model, spaCy.
    """
    services: dict[str, str] = {}

    # ── MongoDB ────────────────────────────────────────────────────────────────
    try:
        from app.db.database import db_instance
        if db_instance.client:
            await db_instance.client.admin.command("ping")
            services["mongodb"] = "healthy"
        else:
            services["mongodb"] = "not connected"
    except Exception as e:
        services["mongodb"] = f"error: {str(e)[:60]}"

    # ── FAISS ──────────────────────────────────────────────────────────────────
    try:
        import faiss
        services["faiss"] = "healthy"
    except ImportError:
        services["faiss"] = "not installed"

    # ── Embedding Model ────────────────────────────────────────────────────────
    try:
        from app.services.embedding.embedding_service import _model
        services["embedding_model"] = (
            f"loaded ({settings.embedding_model})" if _model else "not loaded yet"
        )
    except Exception:
        services["embedding_model"] = "unknown"

    # ── spaCy ──────────────────────────────────────────────────────────────────
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        services["spacy"] = f"healthy ({nlp.meta.get('name', 'en_core_web_sm')})"
    except Exception as e:
        services["spacy"] = f"error: {str(e)[:60]}"

    overall = "healthy" if all(
        "error" not in v and "not installed" not in v
        for v in services.values()
    ) else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        services=services,
    )