"""
app/services/nlp/portfolio_scorer.py
──────────────────────────────────────
Generates:
  • portfolio_score         (0–100)  — overall quality signal
  • technical_depth_score   (0–100)  — depth & sophistication of tech used
  • project_realism_score   (0–100)  — are projects real and meaningful?

Also verifies which skills are actually demonstrated vs just listed.
"""

import re
from loguru import logger


# ── Indicators of technical depth ─────────────────────────────────────────────

DEPTH_INDICATORS = [
    # Architecture & Design
    "system design", "architecture", "microservices", "monolith",
    "design pattern", "solid principles", "mvc", "mvp", "mvvm",
    "clean architecture", "domain-driven", "event-driven",
    # Performance & Scale
    "performance optimization", "scalability", "load balancing",
    "caching", "redis", "cdn", "latency", "throughput",
    "millions of requests", "high traffic",
    # Security
    "authentication", "authorization", "oauth", "jwt", "ssl", "tls",
    "encryption", "security", "penetration testing",
    # Testing
    "unit test", "integration test", "end-to-end", "test coverage",
    "ci/cd", "continuous integration", "automated testing",
    # Data & ML
    "machine learning", "deep learning", "neural network", "model training",
    "data pipeline", "etl", "data warehouse", "analytics",
    # DevOps
    "docker", "kubernetes", "terraform", "infrastructure as code",
    "monitoring", "logging", "alerting", "sre",
    # Database
    "database optimization", "indexing", "query optimization",
    "replication", "sharding", "nosql", "acid",
    # API Design
    "rest api", "graphql", "grpc", "api gateway", "rate limiting",
    "versioning", "swagger", "openapi",
]

# ── Indicators of real / meaningful projects ───────────────────────────────────

REALISM_POSITIVE = [
    "deployed", "production", "live", "users", "clients", "company",
    "startup", "freelance", "github.com", "open source",
    "app store", "play store", "published", "launched",
    "revenue", "customers", "team", "led", "managed",
    "contributed to", "pull request", "open-source",
]

REALISM_NEGATIVE = [
    "todo app", "hello world", "tutorial", "course project",
    "homework", "exercise", "practice project", "sample",
    "demo app", "learning project", "beginner project",
    "bootcamp project", "clone of", "copy of",
]

# ── Demonstration keywords (skill is being actively used) ─────────────────────

DEMONSTRATION_VERBS = [
    "built", "developed", "implemented", "designed", "created",
    "deployed", "architected", "led", "maintained", "optimized",
    "integrated", "migrated", "refactored", "launched", "shipped",
]


def score_technical_depth(text: str, skills: list[dict]) -> float:
    """
    Score 0–100 based on:
      • Depth indicators found in text (max 60 pts)
      • Number and diversity of skills (max 25 pts)
      • Average skill confidence (max 15 pts)
    """
    text_lower = text.lower()

    # Depth indicator hits
    hits = sum(1 for ind in DEPTH_INDICATORS if ind in text_lower)
    depth_pts = min(hits * 3.5, 60)

    # Skill diversity
    categories = {s.get("category") for s in skills}
    diversity_pts = min(len(categories) * 3, 25)

    # Average confidence
    if skills:
        avg_conf = sum(s["confidence"] for s in skills) / len(skills)
    else:
        avg_conf = 0
    conf_pts = avg_conf * 15

    raw = depth_pts + diversity_pts + conf_pts
    return round(min(raw, 100), 1)


def score_project_realism(
    text: str,
    project_count: int,
    experience_level: str,
) -> float:
    """
    Score 0–100 based on:
      • Positive realism signals (max 50 pts)
      • Negative signals deduct pts
      • Project count bonus (max 20 pts)
      • Experience level prior
    """
    text_lower = text.lower()

    positive_hits = sum(1 for ind in REALISM_POSITIVE if ind in text_lower)
    negative_hits = sum(1 for ind in REALISM_NEGATIVE if ind in text_lower)

    pos_pts = min(positive_hits * 5, 50)
    neg_penalty = min(negative_hits * 8, 30)

    # Project count
    project_pts = min(project_count * 4, 20)

    # Experience prior
    level_pts = {"Beginner": 5, "Intermediate": 15, "Expert": 25}.get(
        experience_level, 10
    )

    raw = pos_pts - neg_penalty + project_pts + level_pts
    return round(max(min(raw, 100), 0), 1)


def verify_demonstrated_skills(
    text: str,
    skills: list[dict],
) -> list[str]:
    """
    A skill is 'verified' if it appears alongside a demonstration verb
    within a reasonable window (sentence-level proximity).
    """
    text_lower = text.lower()
    verified = []

    for skill in skills:
        skill_name = skill["name"].lower()

        # Check each sentence for skill + action verb co-occurrence
        sentences = re.split(r"[.\n!?;]", text_lower)
        for sentence in sentences:
            if skill_name in sentence:
                for verb in DEMONSTRATION_VERBS:
                    if verb in sentence:
                        verified.append(skill["name"])
                        break
                break

        # High-confidence skills from GitHub/deployment context auto-verify
        if skill["confidence"] >= 0.90 and "github" in text_lower:
            if skill["name"] not in verified:
                verified.append(skill["name"])

    return list(set(verified))


def compute_spam_penalty(text: str, skills: list[dict]) -> float:
    """
    Detect skill-stuffing / spam profiles.
    Returns penalty 0.0 (clean) → 1.0 (clear spam).
    """
    # Spam signal: too many skills with low confidence
    if len(skills) > 40:
        low_conf = sum(1 for s in skills if s["confidence"] < 0.60)
        if low_conf / len(skills) > 0.6:
            return 0.6

    # Spam signal: very short text with many skills listed
    if len(text) < 500 and len(skills) > 20:
        return 0.5

    return 0.0


def compute_portfolio_score(
    technical_depth: float,
    project_realism: float,
    skill_count: int,
    verified_count: int,
    years: int,
    spam_penalty: float,
) -> float:
    """
    Final portfolio score 0–100:
      • 40% technical depth
      • 30% project realism
      • 15% verified skills ratio
      • 15% experience (years)
      minus spam penalty
    """
    # Verified skill ratio (0–1)
    verified_ratio = verified_count / skill_count if skill_count > 0 else 0

    # Years contribution (cap at 10 years → full 100 pts)
    year_score = min(years * 10, 100)

    raw = (
        0.40 * technical_depth
        + 0.30 * project_realism
        + 0.15 * (verified_ratio * 100)
        + 0.15 * year_score
    )

    # Apply spam penalty
    raw = raw * (1 - spam_penalty)

    return round(max(min(raw, 100), 0), 1)


def analyze_portfolio_quality(
    preprocessed: dict,
    skills: list[dict],
    experience: dict,
) -> dict:
    """
    Master scorer. Calls all sub-scorers and returns consolidated results.

    Args:
        preprocessed: from preprocessor.preprocess_portfolio_text()
        skills:       from skill_extractor.extract_skills()
        experience:   from experience_detector.analyze_experience_and_complexity()
    """
    text = preprocessed.get("cleaned_text", "")

    tech_depth = score_technical_depth(text, skills)
    realism = score_project_realism(
        text,
        experience.get("project_count", 0),
        experience.get("experience_level", "Beginner"),
    )
    verified_skills = verify_demonstrated_skills(text, skills)
    spam_penalty = compute_spam_penalty(text, skills)
    portfolio_score = compute_portfolio_score(
        tech_depth,
        realism,
        len(skills),
        len(verified_skills),
        experience.get("years_of_experience", 0),
        spam_penalty,
    )

    logger.debug(
        f"Portfolio score: {portfolio_score} | Depth: {tech_depth} "
        f"| Realism: {realism} | Verified: {len(verified_skills)}"
    )

    return {
        "portfolio_score": portfolio_score,
        "technical_depth_score": tech_depth,
        "project_realism_score": realism,
        "verified_skills": verified_skills,
        "spam_penalty": spam_penalty,
    }