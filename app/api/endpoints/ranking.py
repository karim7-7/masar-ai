"""
app/api/endpoints/ranking.py
──────────────────────────────
POST /rank-freelancer  — Compute an intelligent ranking score
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.schemas.schemas import RankingInput, RankingResponse
from app.services.ranking.ranking_engine import rank_freelancer
from app.db.database import get_db

router = APIRouter()


@router.post(
    "/rank-freelancer",
    response_model=RankingResponse,
    status_code=status.HTTP_200_OK,
    summary="Rank a freelancer with weighted AI scoring",
    description=(
        "Computes a final ranking score using: Skill Relevance (40%), "
        "Portfolio Quality (25%), Client Ratings (20%), Response Speed (15%). "
        "Includes spam detection and beginner fairness logic."
    ),
    tags=["Ranking"],
)
async def rank_freelancer_endpoint(
    data: RankingInput,
    db=Depends(get_db),
) -> RankingResponse:
    """
    Intelligent freelancer ranking.

    **Inputs** (sent by Node.js backend after aggregating from its DB):
    - `skill_relevance_raw` — pre-computed relevance score 0–100
    - `portfolio_score`     — from portfolio analysis
    - `avg_client_rating`  — average star rating (0–5)
    - `response_time_hours`— average first response time in hours
    - `completed_projects` — total completed jobs on platform
    - `experience_level`   — Beginner / Intermediate / Expert

    **Returns:**
    - `final_score` 0–100
    - `component_scores` — breakdown per factor
    - `ranking_reasons` — human-readable explanations
    """
    logger.info(f"Ranking request — freelancer_id={data.freelancer_id}")

    try:
        result = await rank_freelancer(data, db=db)
        logger.info(
            f"Ranking complete — {data.freelancer_id}: score={result.final_score}"
        )
        return result

    except Exception as e:
        logger.exception(f"Ranking failed for {data.freelancer_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ranking error: {str(e)}",
        )