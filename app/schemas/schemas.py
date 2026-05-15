"""
app/schemas/schemas.py
───────────────────────
Pydantic v2 request & response models for all API endpoints.
These are the contracts between the Node.js backend and this microservice.
"""

from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field, HttpUrl


# ══════════════════════════════════════════════════════════════════════════════
# SHARED
# ══════════════════════════════════════════════════════════════════════════════

class SkillScore(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    verified: bool = False
    source: str = "nlp"


class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, str]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 1 — PORTFOLIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioAnalysisRequest(BaseModel):
    """
    Sent by Node.js backend when a freelancer submits their portfolio.
    At least one of pdf_base64 / portfolio_text / github_url must be provided.
    """
    freelancer_id: str = Field(..., description="Unique freelancer ID from MongoDB")
    pdf_base64: Optional[str] = Field(
        None, description="Base64-encoded PDF bytes of the CV"
    )
    portfolio_text: Optional[str] = Field(
        None, description="Raw portfolio/bio text"
    )
    github_url: Optional[str] = Field(
        None, description="GitHub profile or repo URL"
    )
    portfolio_url: Optional[str] = Field(
        None, description="External portfolio website URL"
    )


class PortfolioAnalysisResponse(BaseModel):
    freelancer_id: str
    skills: list[SkillScore]
    verified_skills: list[str]
    experience_level: Literal["Beginner", "Intermediate", "Expert"]
    years_of_experience: int
    project_complexity: Literal["Simple", "Medium", "Advanced"]
    portfolio_score: float = Field(ge=0, le=100)
    technical_depth_score: float = Field(ge=0, le=100)
    project_realism_score: float = Field(ge=0, le=100)
    analysis_source: list[str]        # e.g. ["pdf", "github"]
    raw_text_length: int
    message: str = "Analysis complete"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 — EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

class GenerateEmbeddingRequest(BaseModel):
    entity_id: str
    entity_type: Literal["freelancer", "project"]
    text: str = Field(..., min_length=10)


class GenerateEmbeddingResponse(BaseModel):
    entity_id: str
    entity_type: str
    embedding_dimension: int
    faiss_index_id: int
    message: str = "Embedding stored"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 — SMART MATCHING
# ══════════════════════════════════════════════════════════════════════════════

class ProjectRequirements(BaseModel):
    project_id: str
    title: str
    description: str
    required_skills: list[str] = []
    budget_range: Optional[str] = None
    complexity: Optional[Literal["Simple", "Medium", "Advanced"]] = None
    experience_required: Optional[Literal["Beginner", "Intermediate", "Expert"]] = None


class FreelancerMatchResult(BaseModel):
    freelancer_id: str
    match_score: float = Field(ge=0, le=100)
    matching_skills: list[str]
    missing_skills: list[str]
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    skill_overlap_score: float = Field(ge=0.0, le=1.0)
    reason: str


class MatchProjectResponse(BaseModel):
    project_id: str
    total_candidates: int
    recommended_freelancers: list[FreelancerMatchResult]
    message: str = "Matching complete"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 3 — RANKING
# ══════════════════════════════════════════════════════════════════════════════

class RankingInput(BaseModel):
    """
    Raw data sent from Node.js for ranking a single freelancer.
    Node.js aggregates this from its own DB before calling us.
    """
    freelancer_id: str
    skill_relevance_raw: float = Field(
        ge=0, le=100, description="Pre-computed skill relevance 0-100"
    )
    portfolio_score: float = Field(
        ge=0, le=100, description="From portfolio analysis"
    )
    avg_client_rating: float = Field(
        ge=0, le=5, description="Average star rating 0-5"
    )
    response_time_hours: float = Field(
        ge=0, description="Average response time in hours"
    )
    completed_projects: int = Field(ge=0)
    experience_level: Literal["Beginner", "Intermediate", "Expert"] = "Beginner"
    is_spam_suspected: bool = False


class RankingResponse(BaseModel):
    freelancer_id: str
    final_score: float = Field(ge=0, le=100)
    component_scores: dict[str, float]
    ranking_reasons: list[str]
    is_spam: bool
    beginner_boost_applied: bool
    message: str = "Ranking complete"