"""
main.py
────────
Masar AI Microservice — FastAPI entry point.

Startup sequence:
  1. Configure logging
  2. Connect to MongoDB
  3. Create indexes
  4. Warm-up embedding model
  5. Reload FAISS indexes from MongoDB
  6. Mount API router
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.database import connect_db, close_db, get_db
from app.models.mongo_models import create_indexes
from app.api.router import api_router


# ── Lifespan (replaces deprecated on_event) ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""

    # ── STARTUP ────────────────────────────────────────────────────────────────
    setup_logging()
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")

    # MongoDB
    await connect_db()
    from app.db.database import db_instance
    await create_indexes(db_instance.db)

    # Warm-up embedding model (loads it into memory once)
    logger.info("Warming up embedding model...")
    try:
        from app.services.embedding.embedding_service import get_embedding_model
        get_embedding_model()
        logger.info("Embedding model ready.")
    except Exception as e:
        logger.warning(f"Embedding model warm-up failed: {e}")

    # Reload FAISS indexes from MongoDB
    logger.info("Rebuilding FAISS indexes from MongoDB...")
    try:
        from app.services.embedding.embedding_service import load_indexes_from_mongodb
        await load_indexes_from_mongodb(db_instance.db)
        logger.info("FAISS indexes ready.")
    except Exception as e:
        logger.warning(f"FAISS index reload failed: {e}")

    logger.info(f"✅ {settings.app_name} is ready — http://{settings.host}:{settings.port}")
    logger.info(f"📚 API docs: http://{settings.host}:{settings.port}/docs")

    yield

    # ── SHUTDOWN ───────────────────────────────────────────────────────────────
    logger.info("Shutting down Masar AI...")
    await close_db()
    logger.info("Goodbye 👋")


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description=(
        "AI Microservice for the Masar freelancing platform. "
        "Provides portfolio analysis, smart matching, and freelancer ranking."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Tighten in production (add Node.js backend origin)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(api_router)


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    })


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )