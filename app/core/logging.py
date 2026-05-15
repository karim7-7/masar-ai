"""
app/core/logging.py
────────────────────
Structured logging using loguru.
"""

import sys
import os
from loguru import logger
from app.core.config import settings


def setup_logging() -> None:
    """Configure loguru with console + file sinks."""
    logger.remove()  # Remove default handler

    # ── Console sink ───────────────────────────────────────────────────────────
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # ── File sink ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
    logger.add(
        settings.log_file,
        level=settings.log_level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
    )

    logger.info(f"Logging initialized — level={settings.log_level}")