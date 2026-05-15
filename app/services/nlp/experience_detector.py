"""
app/services/nlp/experience_detector.py
─────────────────────────────────────────
Infers:
  • Experience level  (Beginner / Intermediate / Expert)
  • Years of experience  (integer)
  • Project complexity  (Simple / Medium / Advanced)

Uses a combination of:
  • Rule-based keyword scoring
  • Years found in text
  • Skill depth & count heuristics
  • Portfolio evidence signals
"""

import re
from loguru import logger

from app.services.nlp.skill_dictionary import (
    COMPLEXITY_KEYWORDS,
    EXPERIENCE_PHRASES,
)


# ── Experience Level ───────────────────────────────────────────────────────────

def detect_experience_level(
    text: str,
    years: int,
    skill_count: int,
    avg_skill_confidence: float,
) -> str:
    """
    Determine Beginner / Intermediate / Expert.

    Scoring rubric (0–100):
      • Years-based:        0-1y=10, 2-3y=30, 4-6y=50, 7+y=70
      • Keyword signals:    up to 20 pts
      • Skill breadth:      up to 10 pts (count > 10 → full 10)
    """
    score = 0

    # ── Years-based scoring ────────────────────────────────────────────────────
    if years >= 7:
        score += 70
    elif years >= 4:
        score += 50
    elif years >= 2:
        score += 30
    elif years >= 1:
        score += 10

    # ── Keyword signals ────────────────────────────────────────────────────────
    text_lower = text.lower()
    for phrase, weight in EXPERIENCE_PHRASES.items():
        if phrase in text_lower:
            score += weight * 4       # Scale phrase weight to pts

    # ── Skill signals ──────────────────────────────────────────────────────────
    if skill_count >= 15:
        score += 10
    elif skill_count >= 8:
        score += 6
    elif skill_count >= 4:
        score += 3

    # ── Advanced tech mentions ─────────────────────────────────────────────────
    advanced_tech = [
        "kubernetes", "microservices", "distributed systems",
        "machine learning", "deep learning", "system design",
        "architecture", "ci/cd", "devops", "terraform",
    ]
    tech_count = sum(1 for t in advanced_tech if t in text_lower)
    score += min(tech_count * 3, 15)

    # ── Map score to level ─────────────────────────────────────────────────────
    if score >= 65:
        return "Expert"
    elif score >= 35:
        return "Intermediate"
    else:
        return "Beginner"


# ── Project Complexity ─────────────────────────────────────────────────────────

def detect_project_complexity(
    text: str,
    skills: list[dict],
    experience_level: str,
) -> str:
    """
    Determine Simple / Medium / Advanced complexity.

    Approach:
      1. Count advanced/medium/simple keyword hits
      2. Bonus for DevOps / cloud / ML skill presence
      3. Use experience level as a prior
    """
    text_lower = text.lower()

    advanced_hits = sum(
        1 for kw in COMPLEXITY_KEYWORDS["advanced"] if kw in text_lower
    )
    medium_hits = sum(
        1 for kw in COMPLEXITY_KEYWORDS["medium"] if kw in text_lower
    )
    simple_hits = sum(
        1 for kw in COMPLEXITY_KEYWORDS["simple"] if kw in text_lower
    )

    # Count high-complexity skill categories present
    complex_categories = {
        "devops_tools", "cloud_platforms", "ai_ml_tools", "blockchain_web3"
    }
    complex_skill_count = sum(
        1 for s in skills if s.get("category") in complex_categories
    )
    advanced_hits += complex_skill_count

    # Decide
    if advanced_hits >= 3 or (advanced_hits >= 2 and experience_level == "Expert"):
        return "Advanced"
    elif medium_hits >= 2 or advanced_hits >= 1:
        return "Medium"
    else:
        return "Simple"


# ── Project Count Estimation ───────────────────────────────────────────────────

def estimate_project_count(text: str) -> int:
    """
    Estimate number of projects mentioned in the portfolio.
    Looks for project headers / numbered lists / project name patterns.
    """
    patterns = [
        r"project\s*\d+",
        r"^\s*\d+\.\s+[A-Z]",           # Numbered list items
        r"##\s+\w+",                      # Markdown headers
        r"project(?:s)?:",
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))

    # Cap at reasonable range
    return min(max(count, 0), 30)


# ── Compound Analysis ─────────────────────────────────────────────────────────

def analyze_experience_and_complexity(
    preprocessed: dict,
    skills: list[dict],
) -> dict:
    """
    Master function combining all detectors.

    Args:
        preprocessed: Output from preprocessor.preprocess_portfolio_text()
        skills:       Output from skill_extractor.extract_skills()

    Returns dict with:
        experience_level, years_of_experience, project_complexity,
        project_count, experience_score (internal 0-100)
    """
    text = preprocessed.get("cleaned_text", "")
    years = preprocessed.get("years_of_experience", 0)
    skill_count = len(skills)
    avg_conf = (
        sum(s["confidence"] for s in skills) / skill_count
        if skill_count > 0
        else 0.0
    )

    experience_level = detect_experience_level(text, years, skill_count, avg_conf)
    project_complexity = detect_project_complexity(text, skills, experience_level)
    project_count = estimate_project_count(text)

    logger.debug(
        f"Experience: {experience_level} | Complexity: {project_complexity} "
        f"| Years: {years} | Projects: {project_count}"
    )

    return {
        "experience_level": experience_level,
        "years_of_experience": years,
        "project_complexity": project_complexity,
        "project_count": project_count,
    }