"""
app/services/nlp/portfolio_analyzer.py
────────────────────────────────────────
Orchestrates the full portfolio analysis pipeline:
  PDF extraction → Preprocessing → Skill extraction
  → Experience detection → Scoring → GitHub verification
  → Structured JSON output
"""

from loguru import logger
from app.schemas.schemas import PortfolioAnalysisRequest, PortfolioAnalysisResponse, SkillScore

from app.services.nlp.pdf_extractor import extract_text_from_base64_pdf
from app.services.nlp.preprocessor import preprocess_portfolio_text
from app.services.nlp.skill_extractor import extract_skills
from app.services.nlp.experience_detector import analyze_experience_and_complexity
from app.services.nlp.portfolio_scorer import analyze_portfolio_quality
from app.services.nlp.github_analyzer import analyze_github_url


async def analyze_portfolio(
    request: PortfolioAnalysisRequest,
    db=None,
) -> PortfolioAnalysisResponse:
    """
    Full portfolio analysis pipeline.

    1. Collect text from all available sources (PDF, text, GitHub, URL)
    2. Preprocess the combined text
    3. Extract skills
    4. Detect experience & complexity
    5. Score portfolio quality
    6. Merge GitHub-verified skills
    7. Persist to MongoDB
    8. Return structured response
    """
    all_text_parts: list[str] = []
    analysis_sources: list[str] = []
    github_verified_skills: list[str] = []

    # ── Step 1: Collect text ──────────────────────────────────────────────────

    # 1a. PDF CV
    if request.pdf_base64:
        try:
            text, method = extract_text_from_base64_pdf(request.pdf_base64)
            all_text_parts.append(text)
            analysis_sources.append(f"pdf:{method}")
            logger.info(f"PDF extracted via {method}: {len(text)} chars")
        except Exception as e:
            logger.error(f"PDF extraction failed for {request.freelancer_id}: {e}")

    # 1b. Portfolio description text
    if request.portfolio_text:
        all_text_parts.append(request.portfolio_text)
        analysis_sources.append("manual_text")

    # 1c. GitHub profile/repo
    if request.github_url:
        try:
            github_result = analyze_github_url(request.github_url)
            if github_result.get("success"):
                # Add README / description text for NLP
                if github_result.get("readme_text"):
                    all_text_parts.append(github_result["readme_text"])
                if github_result.get("description"):
                    all_text_parts.append(github_result["description"])

                # Collect verified skills from real code
                github_verified_skills = github_result.get("verified_skills", [])
                analysis_sources.append("github")
                logger.info(f"GitHub analysis: {len(github_verified_skills)} verified skills")
            else:
                logger.warning(f"GitHub analysis skipped: {github_result.get('error')}")
        except Exception as e:
            logger.warning(f"GitHub analysis error: {e}")

    # Guard: ensure we have at least some text
    if not all_text_parts:
        logger.warning(f"No text sources for freelancer {request.freelancer_id}")
        return _empty_response(request.freelancer_id)

    # ── Step 2: Preprocess combined text ──────────────────────────────────────
    combined_text = "\n\n".join(all_text_parts)
    preprocessed = preprocess_portfolio_text(combined_text)

    # ── Step 3: Extract skills ────────────────────────────────────────────────
    skills = extract_skills(preprocessed["cleaned_text"])

    # Boost confidence for GitHub-verified skills
    github_lower = {s.lower() for s in github_verified_skills}
    for skill in skills:
        if skill["name"].lower() in github_lower:
            skill["confidence"] = min(skill["confidence"] + 0.07, 0.99)
            skill["verified"] = True
            skill["source"] = "github+nlp"

    # Add any GitHub skills not found by NLP
    existing_skill_names = {s["name"].lower() for s in skills}
    for gh_skill in github_verified_skills:
        if gh_skill.lower() not in existing_skill_names:
            skills.append({
                "name": gh_skill,
                "confidence": 0.88,
                "category": "programming_languages",
                "verified": True,
                "source": "github",
            })

    # ── Step 4: Detect experience & complexity ────────────────────────────────
    experience = analyze_experience_and_complexity(preprocessed, skills)

    # ── Step 5: Score portfolio quality ──────────────────────────────────────
    quality = analyze_portfolio_quality(preprocessed, skills, experience)

    # Merge all verified skills
    all_verified = list(set(
        quality["verified_skills"] + github_verified_skills
    ))

    # ── Step 6: Persist to MongoDB ────────────────────────────────────────────
    if db is not None:
        try:
            await _persist_analysis(
                db, request.freelancer_id, skills, experience, quality,
                all_verified, analysis_sources, preprocessed
            )
        except Exception as e:
            logger.error(f"Failed to persist portfolio analysis: {e}")

    # ── Step 7: Build response ────────────────────────────────────────────────
    skill_scores = [
        SkillScore(
            name=s["name"],
            confidence=s["confidence"],
            verified=s.get("verified", False),
            source=s.get("source", "nlp"),
        )
        for s in skills[:30]         # Return top 30 skills
    ]

    return PortfolioAnalysisResponse(
        freelancer_id=request.freelancer_id,
        skills=skill_scores,
        verified_skills=all_verified,
        experience_level=experience["experience_level"],
        years_of_experience=experience["years_of_experience"],
        project_complexity=experience["project_complexity"],
        portfolio_score=quality["portfolio_score"],
        technical_depth_score=quality["technical_depth_score"],
        project_realism_score=quality["project_realism_score"],
        analysis_source=analysis_sources,
        raw_text_length=preprocessed["char_count"],
        message="Analysis complete",
    )


async def _persist_analysis(
    db, freelancer_id, skills, experience, quality,
    verified_skills, sources, preprocessed
):
    """Save analysis results to MongoDB."""
    from app.models.mongo_models import portfolio_analysis_doc, skill_score_doc
    from datetime import datetime

    # Upsert portfolio analysis
    doc = portfolio_analysis_doc(
        freelancer_id=freelancer_id,
        skills=[{"name": s["name"], "confidence": s["confidence"]} for s in skills],
        verified_skills=verified_skills,
        experience_level=experience["experience_level"],
        years_of_experience=experience["years_of_experience"],
        project_complexity=experience["project_complexity"],
        portfolio_score=quality["portfolio_score"],
        technical_depth_score=quality["technical_depth_score"],
        project_realism_score=quality["project_realism_score"],
        raw_text_length=preprocessed["char_count"],
        source=",".join(sources),
    )

    await db["portfolio_analyses"].replace_one(
        {"freelancer_id": freelancer_id},
        doc,
        upsert=True,
    )

    # Upsert individual skill scores
    for skill in skills:
        skill_doc = skill_score_doc(
            freelancer_id=freelancer_id,
            skill_name=skill["name"],
            confidence=skill["confidence"],
            source=skill.get("source", "nlp"),
            verified=skill.get("verified", False),
        )
        await db["skill_scores"].update_one(
            {"freelancer_id": freelancer_id, "skill_name": skill["name"]},
            {"$set": skill_doc},
            upsert=True,
        )


def _empty_response(freelancer_id: str) -> PortfolioAnalysisResponse:
    """Return a zero-value response when no text is available."""
    return PortfolioAnalysisResponse(
        freelancer_id=freelancer_id,
        skills=[],
        verified_skills=[],
        experience_level="Beginner",
        years_of_experience=0,
        project_complexity="Simple",
        portfolio_score=0.0,
        technical_depth_score=0.0,
        project_realism_score=0.0,
        analysis_source=[],
        raw_text_length=0,
        message="No analyzable content provided",
    )