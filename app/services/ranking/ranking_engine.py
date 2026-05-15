"""
app/services/ranking/ranking_engine.py
────────────────────────────────────────
Ranking Engine:
  Weights: Skill Relevance 40% | Portfolio Quality 25%
           Client Ratings 20% | Response Speed 15%

Special logic:
  • Spam profile penalization
  • Beginner fairness boost
  • Explainable ranking reasons
  • Normalized 0–100 output
"""

from loguru import logger
from app.core.config import settings
from app.schemas.schemas import RankingInput, RankingResponse


# ── Normalization Helpers ─────────────────────────────────────────────────────

def normalize_rating(raw_rating: float, scale: float = 5.0) -> float:
    """Convert 0–5 star rating to 0–100 score."""
    return round((raw_rating / scale) * 100, 1)


def normalize_response_time(hours: float) -> float:
    """
    Convert average response time (hours) to 0–100 score.
    Faster response = higher score.
      < 1h  → 100
      1–4h  → 90
      4–12h → 70
      12–24h→ 50
      > 48h → 20
      > 7d  → 0
    """
    if hours < 1:
        return 100.0
    elif hours < 4:
        return 90.0
    elif hours < 12:
        return 70.0
    elif hours < 24:
        return 50.0
    elif hours < 48:
        return 35.0
    elif hours < 168:  # 1 week
        return 20.0
    else:
        return 5.0


# ── Spam Detection ────────────────────────────────────────────────────────────

def detect_spam_signals(data: RankingInput) -> tuple[bool, float]:
    """
    Returns (is_spam, penalty_multiplier).
    penalty_multiplier 1.0 = no penalty, 0.0 = fully penalized.
    """
    if data.is_spam_suspected:
        return True, 0.20   # Hard penalization

    signals = 0

    # Zero activity with claimed high rating
    if data.completed_projects == 0 and data.avg_client_rating > 4.5:
        signals += 2

    # Extremely high skill relevance but zero completed projects
    if data.skill_relevance_raw > 90 and data.completed_projects == 0:
        signals += 1

    # Suspiciously perfect scores across the board
    if (
        data.skill_relevance_raw >= 98
        and data.portfolio_score >= 98
        and data.avg_client_rating >= 4.9
        and data.response_time_hours <= 0.5
    ):
        signals += 3

    if signals >= 3:
        return True, 0.30
    elif signals >= 2:
        return False, 0.70
    else:
        return False, 1.0


# ── Beginner Fairness ─────────────────────────────────────────────────────────

def apply_beginner_fairness(
    base_score: float,
    data: RankingInput,
) -> tuple[float, bool]:
    """
    Boost highly skilled beginners so they can compete.

    Conditions for boost:
      • experience_level == "Beginner"
      • portfolio_score >= 70 (strong portfolio)
      • skill_relevance_raw >= 65

    Boost: +10 pts (capped at 100)
    """
    if (
        data.experience_level == "Beginner"
        and data.portfolio_score >= 70
        and data.skill_relevance_raw >= 65
    ):
        boosted = min(base_score + 10.0, 100.0)
        logger.debug(f"Beginner boost applied: {base_score:.1f} → {boosted:.1f}")
        return boosted, True

    return base_score, False


# ── Reason Generation ─────────────────────────────────────────────────────────

def generate_ranking_reasons(
    data: RankingInput,
    component_scores: dict[str, float],
    is_spam: bool,
    beginner_boost: bool,
) -> list[str]:
    """Generate a list of human-readable ranking explanations."""
    reasons = []

    # Skill relevance
    skill_score = component_scores["skill_relevance"]
    if skill_score >= 85:
        reasons.append("Highly relevant skill set for this domain")
    elif skill_score >= 70:
        reasons.append("Good skill alignment with job requirements")
    elif skill_score < 40:
        reasons.append("Limited skill relevance detected")

    # Portfolio quality
    port_score = component_scores["portfolio_quality"]
    if port_score >= 85:
        reasons.append("Excellent portfolio quality with verified projects")
    elif port_score >= 70:
        reasons.append("Strong portfolio demonstrates practical experience")
    elif port_score < 40:
        reasons.append("Portfolio needs improvement")

    # Client rating
    rating_score = component_scores["client_rating"]
    if rating_score >= 90:
        reasons.append(f"Outstanding client rating ({data.avg_client_rating:.1f}/5.0)")
    elif rating_score >= 70:
        reasons.append(f"Good client satisfaction ({data.avg_client_rating:.1f}/5.0)")
    elif data.completed_projects == 0:
        reasons.append("No completed projects yet — new freelancer")

    # Response speed
    resp_score = component_scores["response_speed"]
    if resp_score >= 90:
        reasons.append("Excellent response rate (responds within 1 hour)")
    elif resp_score >= 70:
        reasons.append("Good response speed")
    elif resp_score < 40:
        reasons.append("Slow response time may affect client experience")

    # Completed projects
    if data.completed_projects >= 20:
        reasons.append(f"Highly experienced with {data.completed_projects} completed projects")
    elif data.completed_projects >= 5:
        reasons.append(f"{data.completed_projects} completed projects on platform")

    # Spam
    if is_spam:
        reasons.append("⚠️ Profile flagged for review — score penalized")

    # Beginner boost
    if beginner_boost:
        reasons.append("🌟 Rising talent — strong portfolio boosts score")

    return reasons


# ── Main Ranking Function ──────────────────────────────────────────────────────

async def rank_freelancer(data: RankingInput, db=None) -> RankingResponse:
    """
    Compute final ranking score for a single freelancer.

    Steps:
    1. Normalize each input metric to 0–100
    2. Detect spam signals
    3. Compute weighted sum
    4. Apply spam penalty
    5. Apply beginner fairness boost
    6. Generate reasons
    7. Persist to MongoDB
    """

    # ── Step 1: Normalize all components to 0–100 ──────────────────────────────
    skill_score = float(max(0, min(data.skill_relevance_raw, 100)))
    portfolio_score = float(max(0, min(data.portfolio_score, 100)))
    rating_score = normalize_rating(data.avg_client_rating)
    response_score = normalize_response_time(data.response_time_hours)

    component_scores = {
        "skill_relevance": round(skill_score, 1),
        "portfolio_quality": round(portfolio_score, 1),
        "client_rating": round(rating_score, 1),
        "response_speed": round(response_score, 1),
    }

    # ── Step 2: Spam detection ─────────────────────────────────────────────────
    is_spam, penalty_multiplier = detect_spam_signals(data)

    # ── Step 3: Weighted sum ───────────────────────────────────────────────────
    weights = {
        "skill_relevance": settings.weight_skill_relevance,
        "portfolio_quality": settings.weight_portfolio_quality,
        "client_rating": settings.weight_client_rating,
        "response_speed": settings.weight_response_speed,
    }

    raw_score = sum(
        component_scores[k] * weights[k]
        for k in weights
    )

    # ── Step 4: Apply spam penalty ────────────────────────────────────────────
    penalized_score = raw_score * penalty_multiplier

    # ── Step 5: Beginner fairness boost ───────────────────────────────────────
    final_score, beginner_boost = apply_beginner_fairness(penalized_score, data)
    final_score = round(max(0.0, min(final_score, 100.0)), 1)

    # ── Step 6: Generate reasons ───────────────────────────────────────────────
    reasons = generate_ranking_reasons(data, component_scores, is_spam, beginner_boost)

    # ── Step 7: Persist ───────────────────────────────────────────────────────
    if db is not None:
        try:
            await _persist_ranking(
                db, data.freelancer_id, final_score, component_scores,
                reasons, is_spam, beginner_boost
            )
        except Exception as e:
            logger.error(f"Failed to persist ranking: {e}")

    logger.info(
        f"Ranked {data.freelancer_id}: {final_score}/100 "
        f"(spam={is_spam}, beginner_boost={beginner_boost})"
    )

    return RankingResponse(
        freelancer_id=data.freelancer_id,
        final_score=final_score,
        component_scores=component_scores,
        ranking_reasons=reasons,
        is_spam=is_spam,
        beginner_boost_applied=beginner_boost,
        message="Ranking complete",
    )


async def _persist_ranking(
    db, freelancer_id, final_score, component_scores,
    reasons, is_spam, beginner_boost
):
    from app.models.mongo_models import ranking_score_doc
    doc = ranking_score_doc(
        freelancer_id=freelancer_id,
        final_score=final_score,
        skill_relevance_score=component_scores["skill_relevance"],
        portfolio_quality_score=component_scores["portfolio_quality"],
        client_rating_score=component_scores["client_rating"],
        response_speed_score=component_scores["response_speed"],
        ranking_reasons=reasons,
        is_spam=is_spam,
        beginner_boost=beginner_boost,
    )
    await db["ranking_scores"].replace_one(
        {"freelancer_id": freelancer_id},
        doc,
        upsert=True,
    )