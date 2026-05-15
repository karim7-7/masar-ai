"""
app/api/endpoints/matching.py
───────────────────────────────
POST /generate-embedding   — Store a freelancer/project embedding
POST /match-project        — Find best-matching freelancers for a project
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.schemas.schemas import (
    GenerateEmbeddingRequest,
    GenerateEmbeddingResponse,
    ProjectRequirements,
    MatchProjectResponse,
)
from app.services.embedding.embedding_service import (
    store_embedding,
    build_freelancer_text,
    build_project_text,
)
from app.services.matching.matching_engine import match_project_to_freelancers
from app.db.database import get_db

router = APIRouter()


# ── POST /generate-embedding ──────────────────────────────────────────────────

@router.post(
    "/generate-embedding",
    response_model=GenerateEmbeddingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate and store a text embedding",
    description=(
        "Generates a sentence-transformer embedding for a freelancer or project "
        "and stores it in FAISS + MongoDB for future similarity search."
    ),
    tags=["Embeddings & Matching"],
)
async def generate_embedding_endpoint(
    request: GenerateEmbeddingRequest,
    db=Depends(get_db),
) -> GenerateEmbeddingResponse:
    """
    Generate embedding for any entity (freelancer or project).

    The `text` field should be a rich description including skills, experience,
    and project details to maximize matching quality.
    """
    logger.info(
        f"Embedding request — entity_type={request.entity_type}, "
        f"entity_id={request.entity_id}"
    )

    try:
        embedding, faiss_id = await store_embedding(
            entity_id=request.entity_id,
            entity_type=request.entity_type,
            text=request.text,
            db=db,
        )

        return GenerateEmbeddingResponse(
            entity_id=request.entity_id,
            entity_type=request.entity_type,
            embedding_dimension=len(embedding),
            faiss_index_id=faiss_id,
            message="Embedding stored successfully",
        )

    except Exception as e:
        logger.exception(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding error: {str(e)}",
        )


# ── POST /match-project ───────────────────────────────────────────────────────

@router.post(
    "/match-project",
    response_model=MatchProjectResponse,
    status_code=status.HTTP_200_OK,
    summary="Match a project to best-fit freelancers",
    description=(
        "Combines FAISS semantic search + skill overlap + experience compatibility "
        "to return a ranked list of freelancers for a given project."
    ),
    tags=["Embeddings & Matching"],
)
async def match_project_endpoint(
    project: ProjectRequirements,
    top_k: int = 10,
    db=Depends(get_db),
) -> MatchProjectResponse:
    """
    Smart project-freelancer matching.

    **How it works:**
    1. Project description is embedded using sentence-transformers
    2. FAISS finds top semantic candidates
    3. Each candidate is scored on skill overlap + experience fit + portfolio quality
    4. Returns ranked list with explanations

    **Prerequisites:** Freelancer embeddings must exist (call `/generate-embedding` first)
    """
    logger.info(
        f"Match request — project_id={project.project_id}, top_k={top_k}"
    )

    if top_k < 1 or top_k > 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="top_k must be between 1 and 50",
        )

    try:
        result = await match_project_to_freelancers(project, db, top_k=top_k)
        logger.info(
            f"Matching complete — project={project.project_id}, "
            f"candidates={result.total_candidates}"
        )
        return result

    except Exception as e:
        logger.exception(f"Matching failed for project {project.project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Matching error: {str(e)}",
        )