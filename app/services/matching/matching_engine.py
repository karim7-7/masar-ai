"""
app/services/matching/matching_engine.py
─────────────────────────────────────────
Smart Matching Engine:
  Combines semantic similarity (FAISS) + skill overlap + experience
  to produce a ranked list of freelancer-project matches.

Match score formula:
  40% semantic similarity
  30% skill overlap
  20% experience compatibility
  10% portfolio quality
"""

from loguru import logger
from app.schemas.schemas import (
    ProjectRequirements,
    MatchProjectResponse,
    FreelancerMatchResult,
)
from app.services.embedding.embedding_service import (
    search_similar,
    generate_embedding,
    build_project_text,
)
from app.services.nlp.skill_extractor import calculate_skill_overlap


# ── Weights ────────────────────────────────────────────────────────────────────
W_SEMANTIC = 0.40
W_SKILL = 0.30
W_EXPERIENCE = 0.20
W_PORTFOLIO = 0.10

EXPERIENCE_ORDER = {"Beginner": 0, "Intermediate": 1, "Expert": 2}


async def match_project_to_freelancers(
    project: ProjectRequirements,
    db,
    top_k: int = 10,
) -> MatchProjectResponse:
    """
    Main matching pipeline:
    1. Embed the project description
    2. FAISS search for top candidates
    3. For each candidate, load profile from MongoDB
    4. Compute composite match score
    5. Return ranked results
    """

    # ── Step 1: Build project text + FAISS search ──────────────────────────────
    project_text = build_project_text(project.model_dump())
    candidates = await search_similar(
        query_text=project_text,
        entity_type="freelancer",
        top_k=top_k * 2,   # Over-fetch; we'll re-rank and trim
        db=db,
    )

    if not candidates:
        logger.warning(f"No FAISS candidates found for project {project.project_id}")
        return MatchProjectResponse(
            project_id=project.project_id,
            total_candidates=0,
            recommended_freelancers=[],
            message="No freelancer candidates found — ensure embeddings are generated",
        )

    # ── Step 2: Load freelancer profiles from MongoDB ─────────────────────────
    freelancer_ids = [c["entity_id"] for c in candidates]
    profiles_cursor = db["freelancer_profiles"].find(
        {"freelancer_id": {"$in": freelancer_ids}}
    )
    profiles = {p["freelancer_id"]: p async for p in profiles_cursor}

    # Also load portfolio analyses
    analyses_cursor = db["portfolio_analyses"].find(
        {"freelancer_id": {"$in": freelancer_ids}}
    )
    analyses = {a["freelancer_id"]: a async for a in analyses_cursor}

    # ── Step 3: Score each candidate ─────────────────────────────────────────
    results = []
    for candidate in candidates:
        fid = candidate["entity_id"]
        profile = profiles.get(fid)
        analysis = analyses.get(fid)

        if not profile:
            logger.debug(f"No profile found for freelancer {fid} — skipping")
            continue

        match_result = _compute_match_score(
            project=project,
            freelancer_id=fid,
            profile=profile,
            analysis=analysis,
            semantic_score=candidate["similarity_score"],
        )
        results.append(match_result)

    # ── Step 4: Sort by match_score descending ────────────────────────────────
    results.sort(key=lambda x: x.match_score, reverse=True)
    top_results = results[:top_k]

    # ── Step 5: Persist match results ─────────────────────────────────────────
    try:
        await _persist_match_results(db, project.project_id, top_results)
    except Exception as e:
        logger.error(f"Failed to persist match results: {e}")

    return MatchProjectResponse(
        project_id=project.project_id,
        total_candidates=len(results),
        recommended_freelancers=top_results,
        message=f"Matched {len(top_results)} freelancers",
    )


def _compute_match_score(
    project: ProjectRequirements,
    freelancer_id: str,
    profile: dict,
    analysis: dict | None,
    semantic_score: float,
) -> FreelancerMatchResult:
    """
    Compute a composite match score for one freelancer-project pair.
    All sub-scores normalized to [0, 1] before weighting.
    """

    # ── Semantic similarity (already cosine, 0-1) ──────────────────────────────
    sem_score = max(0.0, min(semantic_score, 1.0))

    # ── Skill overlap ─────────────────────────────────────────────────────────
    freelancer_skills_raw = profile.get("skills", [])
    if freelancer_skills_raw and isinstance(freelancer_skills_raw[0], dict):
        freelancer_skill_names = [s["name"] for s in freelancer_skills_raw]
    else:
        freelancer_skill_names = freelancer_skills_raw

    # Also add NLP-extracted skills from analysis
    if analysis:
        analysis_skills = [
            s["name"] if isinstance(s, dict) else s
            for s in analysis.get("skills", [])
        ]
        freelancer_skill_names = list(set(freelancer_skill_names + analysis_skills))

    matching_skills, missing_skills, skill_overlap = calculate_skill_overlap(
        freelancer_skill_names,
        project.required_skills,
    )

    # ── Experience compatibility ───────────────────────────────────────────────
    exp_score = _experience_compatibility(
        freelancer_level=profile.get("experience_level", "Beginner"),
        required_level=project.experience_required,
    )

    # ── Portfolio quality ─────────────────────────────────────────────────────
    portfolio_quality_raw = 0.0
    if analysis:
        portfolio_quality_raw = analysis.get("portfolio_score", 0.0) / 100.0
    elif profile.get("portfolio_score"):
        portfolio_quality_raw = profile["portfolio_score"] / 100.0

    # ── Composite score ───────────────────────────────────────────────────────
    composite = (
        W_SEMANTIC * sem_score
        + W_SKILL * skill_overlap
        + W_EXPERIENCE * exp_score
        + W_PORTFOLIO * portfolio_quality_raw
    )
    final_score = round(composite * 100, 1)

    # ── Generate human-readable reason ────────────────────────────────────────
    reason = _generate_reason(
        matching_skills, missing_skills, sem_score, exp_score,
        profile.get("experience_level", "Beginner"),
    )

    return FreelancerMatchResult(
        freelancer_id=freelancer_id,
        match_score=final_score,
        matching_skills=matching_skills,
        missing_skills=missing_skills,
        semantic_similarity=round(sem_score, 3),
        skill_overlap_score=round(skill_overlap, 3),
        reason=reason,
    )


def _experience_compatibility(
    freelancer_level: str,
    required_level: str | None,
) -> float:
    """
    Returns 0.0–1.0 compatibility.
    Over-qualified freelancers still score well (just slightly penalized).
    Under-qualified freelancers score lower.
    """
    if not required_level:
        return 0.85       # No requirement → neutral positive

    f_order = EXPERIENCE_ORDER.get(freelancer_level, 0)
    r_order = EXPERIENCE_ORDER.get(required_level, 0)

    diff = f_order - r_order

    if diff == 0:
        return 1.0        # Perfect match
    elif diff == 1:
        return 0.85       # Over-qualified (still good)
    elif diff == 2:
        return 0.65       # Very over-qualified
    elif diff == -1:
        return 0.50       # Slightly under-qualified
    else:
        return 0.20       # Significantly under-qualified


def _generate_reason(
    matching: list[str],
    missing: list[str],
    sem_score: float,
    exp_score: float,
    level: str,
) -> str:
    """Generate a concise human-readable match explanation."""
    parts = []

    if matching:
        skills_str = ", ".join(matching[:3])
        parts.append(f"Matches on {skills_str}")
        if len(matching) > 3:
            parts[-1] += f" and {len(matching) - 3} more skills"

    if sem_score >= 0.75:
        parts.append("strong semantic alignment with project requirements")
    elif sem_score >= 0.55:
        parts.append("moderate semantic alignment")

    if exp_score >= 0.9:
        parts.append(f"{level}-level experience is a great fit")
    elif exp_score >= 0.6:
        parts.append(f"{level}-level experience is adequate")

    if missing:
        parts.append(f"missing: {', '.join(missing[:2])}")

    return ". ".join(parts).capitalize() if parts else "Candidate identified via semantic search"


async def _persist_match_results(db, project_id: str, results: list[FreelancerMatchResult]):
    """Save top match results to MongoDB."""
    from app.models.mongo_models import match_result_doc

    for r in results:
        doc = match_result_doc(
            project_id=project_id,
            freelancer_id=r.freelancer_id,
            match_score=r.match_score,
            matching_skills=r.matching_skills,
            missing_skills=r.missing_skills,
            reason=r.reason,
            semantic_similarity=r.semantic_similarity,
            skill_overlap_score=r.skill_overlap_score,
        )
        await db["match_results"].replace_one(
            {"project_id": project_id, "freelancer_id": r.freelancer_id},
            doc,
            upsert=True,
        )