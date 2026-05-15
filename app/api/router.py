"""
app/api/router.py
──────────────────
Central router — aggregates all endpoint routers.
"""

from fastapi import APIRouter
from app.api.endpoints import portfolio, matching, ranking, health

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(portfolio.router)
api_router.include_router(matching.router)
api_router.include_router(ranking.router)