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
#-------------------------------------------------------------------

def analyze_spam_risk(
    text: str,
    skills: list[dict],
    github_analysis: dict | None = None,
    github_repo_analysis: list[dict] | None = None,
    portfolio_url: str | None = None,
) -> dict:
    """
    Return spam/suspicious profile analysis based on CV text, GitHub, portfolio, and skills.
    """

    reasons = []
    spam_score = 0

    text_lower = (text or "").lower()
    github_repo_analysis = github_repo_analysis or []

    # 1. Very short CV/portfolio text
    if len(text.strip()) < 500:
        spam_score += 20
        reasons.append("Very limited CV/portfolio text was provided")

    # 2. Too many skills compared to text size
    if len(skills) > 25 and len(text.strip()) < 1200:
        spam_score += 20
        reasons.append("Too many claimed skills compared to available evidence")

    # 3. Low confidence skills
    if skills:
        low_conf_count = sum(1 for s in skills if s.get("confidence", 0) < 0.65)
        low_conf_ratio = low_conf_count / len(skills)

        if low_conf_ratio > 0.5:
            spam_score += 15
            reasons.append("Many extracted skills have low confidence")

    # 4. Missing GitHub evidence
    if not github_analysis or not github_analysis.get("success"):
        spam_score += 15
        reasons.append("GitHub profile is missing or could not be verified")
    else:
        public_repos = github_analysis.get("public_repos", 0)
        top_languages = github_analysis.get("top_languages", [])

        if public_repos == 0:
            spam_score += 15
            reasons.append("GitHub profile has no public repositories")

        if not top_languages:
            spam_score += 10
            reasons.append("GitHub profile has no detectable programming languages")

    # 5. Repos exist but weak repo evidence
    successful_repos = [
        repo for repo in github_repo_analysis
        if repo.get("success")
    ]

    if github_analysis and github_analysis.get("success"):
        if not successful_repos:
            spam_score += 10
            reasons.append("No GitHub repositories could be analyzed")

    # 6. Missing portfolio URL
    if not portfolio_url:
        spam_score += 10
        reasons.append("Portfolio URL is missing")

    # 7. Generic / tutorial projects
    generic_keywords = [
        "todo app",
        "hello world",
        "tutorial",
        "course project",
        "practice project",
        "demo app",
        "clone",
    ]

    generic_hits = sum(1 for word in generic_keywords if word in text_lower)

    if generic_hits >= 2:
        spam_score += 15
        reasons.append("Profile contains multiple generic or tutorial project signals")

    spam_score = max(0, min(spam_score, 100))

    if spam_score >= 70:
        risk_level = "high"
    elif spam_score >= 40:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "spam_score": spam_score,
        "is_suspicious": spam_score >= 40,
        "risk_level": risk_level,
        "reasons": reasons,
    }
#-------------------------------------------------------------------



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
    github_analysis: dict | None = None,
    github_repo_analysis: list[dict] | None = None,
    portfolio_url: str | None = None,
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

    spam_check = analyze_spam_risk(
    text=text,
    skills=skills,
    github_analysis=github_analysis,
    github_repo_analysis=github_repo_analysis,
    portfolio_url=portfolio_url,
)


    portfolio_score = compute_portfolio_score(
        tech_depth,
        realism,
        len(skills),
        len(verified_skills),
        experience.get("years_of_experience", 0),
        spam_penalty,
    )

    final_score = round(
    max(
        min(
            portfolio_score - (spam_check["spam_score"] * 0.25),
            100,
        ),
        0,
    ),
    1,
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
    "spam_check": spam_check,
    "final_score": final_score,
}