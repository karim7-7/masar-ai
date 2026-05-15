"""
app/db/database.py
───────────────────
Async MongoDB connection via Motor.
Provides a single client/db instance reused across the app lifecycle.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from loguru import logger
from app.core.config import settings


class Database:
    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None


db_instance = Database()


async def connect_db() -> None:
    """Open MongoDB connection on app startup."""
    logger.info(f"Connecting to MongoDB at {settings.mongodb_uri} ...")
    db_instance.client = AsyncIOMotorClient(settings.mongodb_uri)
    db_instance.db = db_instance.client[settings.mongodb_db_name]

    # Ping to verify connection
    await db_instance.client.admin.command("ping")
    logger.info(f"Connected to MongoDB — database: '{settings.mongodb_db_name}'")


async def close_db() -> None:
    """Close MongoDB connection on app shutdown."""
    if db_instance.client:
        db_instance.client.close()
        logger.info("MongoDB connection closed.")


def get_db() -> AsyncIOMotorDatabase:
    """Dependency: returns the active database instance."""
    if db_instance.db is None:
        raise RuntimeError("Database not initialized. Call connect_db() first.")
    return db_instance.db


# ── Collection helpers ─────────────────────────────────────────────────────────
def get_collection(name: str):
    return get_db()[name]