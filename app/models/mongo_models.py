"""
app/models/mongo_models.py
───────────────────────────
MongoDB collection schemas represented as Python dataclasses / dicts.
These define the document structure stored in each collection.
We use plain dicts + Pydantic for validation rather than a heavy ODM
so the AI microservice stays lightweight and decoupled.

Collections:
  • freelancer_profiles
  • portfolio_analyses
  • skill_scores
  • match_results
  • ranking_scores
  • embeddings
"""

from datetime import datetime
from typing import Any


# ── Index creation helpers ────────────────────────────────────────────────────

INDEXES: dict[str, list[dict]] = {
    "freelancer_profiles": [
        {"key": [("freelancer_id", 1)], "unique": True},
        {"key": [("skills.name", 1)]},
    ],
    "portfolio_analyses": [
        {"key": [("freelancer_id", 1)]},
        {"key": [("created_at", -1)]},
    ],
    "skill_scores": [
        {"key": [("freelancer_id", 1)]},
    ],
    "match_results": [
        {"key": [("project_id", 1)]},
        {"key": [("created_at", -1)]},
    ],
    "ranking_scores": [
        {"key": [("freelancer_id", 1)], "unique": True},
        {"key": [("final_score", -1)]},
    ],
    "embeddings": [
        {"key": [("entity_id", 1), ("entity_type", 1)], "unique": True},
    ],
}


async def create_indexes(db) -> None:
    """Create MongoDB indexes for all collections at startup."""
    from loguru import logger

    for collection_name, indexes in INDEXES.items():
        collection = db[collection_name]
        for index_def in indexes:
            await collection.create_index(
                index_def["key"],
                unique=index_def.get("unique", False),
                background=True,
            )
    logger.info("MongoDB indexes created/verified.")


# ── Document factory functions ─────────────────────────────────────────────────
# These return clean dict templates matching each collection's schema.


def freelancer_profile_doc(
    freelancer_id: str,
    name: str = "",
    email: str = "",
    skills: list[dict] = None,
    experience_level: str = "Beginner",
    years_of_experience: int = 0,
    portfolio_url: str = "",
    github_url: str = "",
    avg_rating: float = 0.0,
    response_rate: float = 0.0,
    completed_projects: int = 0,
) -> dict[str, Any]:
    return {
        "freelancer_id": freelancer_id,
        "name": name,
        "email": email,
        "skills": skills or [],
        "experience_level": experience_level,
        "years_of_experience": years_of_experience,
        "portfolio_url": portfolio_url,
        "github_url": github_url,
        "avg_rating": avg_rating,
        "response_rate": response_rate,
        "completed_projects": completed_projects,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


def portfolio_analysis_doc(
    freelancer_id: str,
    skills: list[dict] = None,
    verified_skills: list[str] = None,
    experience_level: str = "Beginner",
    years_of_experience: int = 0,
    project_complexity: str = "Simple",
    portfolio_score: float = 0.0,
    technical_depth_score: float = 0.0,
    project_realism_score: float = 0.0,
    raw_text_length: int = 0,
    source: str = "pdf",          # pdf | github | url | manual
) -> dict[str, Any]:
    return {
        "freelancer_id": freelancer_id,
        "skills": skills or [],
        "verified_skills": verified_skills or [],
        "experience_level": experience_level,
        "years_of_experience": years_of_experience,
        "project_complexity": project_complexity,
        "portfolio_score": portfolio_score,
        "technical_depth_score": technical_depth_score,
        "project_realism_score": project_realism_score,
        "raw_text_length": raw_text_length,
        "source": source,
        "created_at": datetime.utcnow(),
    }


def skill_score_doc(
    freelancer_id: str,
    skill_name: str,
    confidence: float = 0.0,
    source: str = "nlp",           # nlp | github | manual
    verified: bool = False,
) -> dict[str, Any]:
    return {
        "freelancer_id": freelancer_id,
        "skill_name": skill_name,
        "confidence": confidence,
        "source": source,
        "verified": verified,
        "created_at": datetime.utcnow(),
    }


def match_result_doc(
    project_id: str,
    freelancer_id: str,
    match_score: float = 0.0,
    matching_skills: list[str] = None,
    missing_skills: list[str] = None,
    reason: str = "",
    semantic_similarity: float = 0.0,
    skill_overlap_score: float = 0.0,
) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "freelancer_id": freelancer_id,
        "match_score": match_score,
        "matching_skills": matching_skills or [],
        "missing_skills": missing_skills or [],
        "reason": reason,
        "semantic_similarity": semantic_similarity,
        "skill_overlap_score": skill_overlap_score,
        "created_at": datetime.utcnow(),
    }


def ranking_score_doc(
    freelancer_id: str,
    final_score: float = 0.0,
    skill_relevance_score: float = 0.0,
    portfolio_quality_score: float = 0.0,
    client_rating_score: float = 0.0,
    response_speed_score: float = 0.0,
    ranking_reasons: list[str] = None,
    is_spam: bool = False,
    beginner_boost: bool = False,
) -> dict[str, Any]:
    return {
        "freelancer_id": freelancer_id,
        "final_score": final_score,
        "component_scores": {
            "skill_relevance": skill_relevance_score,
            "portfolio_quality": portfolio_quality_score,
            "client_rating": client_rating_score,
            "response_speed": response_speed_score,
        },
        "ranking_reasons": ranking_reasons or [],
        "is_spam": is_spam,
        "beginner_boost": beginner_boost,
        "updated_at": datetime.utcnow(),
    }


def embedding_doc(
    entity_id: str,
    entity_type: str,       # "freelancer" | "project"
    embedding: list[float] = None,
    faiss_index_id: int = -1,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "embedding": embedding or [],
        "faiss_index_id": faiss_index_id,
        "model_name": model_name,
        "created_at": datetime.utcnow(),
    }