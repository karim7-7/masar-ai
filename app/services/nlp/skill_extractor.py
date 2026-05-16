"""
app/services/nlp/skill_extractor.py
────────────────────────────────────
Multi-strategy skill extraction:
  1. Dictionary lookup  (exact + alias matching)
  2. Regex pattern matching  (handles multi-word, versioned skills)
  3. spaCy NER + noun chunks  (catches unlisted skills)

Returns a list of SkillScore objects with confidence values.
"""

import re
from collections import defaultdict
from loguru import logger

from app.services.nlp.skill_dictionary import (
    SKILL_DICTIONARY,
    SKILL_CANONICAL,
    SKILL_ALIASES,
    ALL_SKILLS,
)


# ── Helper: build multi-word regex patterns ────────────────────────────────────

def _build_skill_patterns() -> list[tuple[re.Pattern, str]]:
    """Compile regex patterns for all skills in the dictionary."""
    patterns = []
    for skills in SKILL_DICTIONARY.values():
        for skill in skills:
            # Escape and build word-boundary pattern
            escaped = re.escape(skill)
            pattern = re.compile(
                rf"\b{escaped}\b",
                re.IGNORECASE
            )
            patterns.append((pattern, skill))

    # Add version-aware patterns (e.g. "Python 3", "Node.js 18")
    version_pattern = re.compile(
        r"\b(Python|Node\.js|React|Vue|Angular|PHP|Java|Go|Ruby|Rust)\s*[\d\.]+\b",
        re.IGNORECASE,
    )
    patterns.append((version_pattern, "__versioned__"))

    return patterns


_SKILL_PATTERNS = _build_skill_patterns()


# ── Strategy 1: Dictionary + Alias Matching ───────────────────────────────────

def _dictionary_match(text: str) -> dict[str, float]:
    """Match skills using compiled regex patterns. Returns {skill: confidence}."""
    found: dict[str, float] = {}
    text_lower = text.lower()

    # Alias resolution first
    for alias, canonical in SKILL_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text_lower):
            found[canonical] = max(found.get(canonical, 0), 0.80)

    # Main dictionary patterns
    for pattern, skill_name in _SKILL_PATTERNS:
        if skill_name == "__versioned__":
            matches = pattern.findall(text)
            for match in matches:
                canonical = SKILL_CANONICAL.get(match.lower(), match)
                found[canonical] = max(found.get(canonical, 0), 0.88)
            continue

        if pattern.search(text):
            found[skill_name] = max(found.get(skill_name, 0), 0.85)

    return found


# ── Strategy 2: Frequency-Based Boost ─────────────────────────────────────────

def _frequency_boost(text: str, found_skills: dict[str, float]) -> dict[str, float]:
    """
    Boost confidence if a skill appears multiple times.
    Repeated mentions → higher confidence (capped at 0.97).
    """
    text_lower = text.lower()
    for skill in list(found_skills.keys()):
        count = len(re.findall(rf"\b{re.escape(skill.lower())}\b", text_lower))
        if count >= 3:
            found_skills[skill] = min(found_skills[skill] + 0.08, 0.97)
        elif count >= 2:
            found_skills[skill] = min(found_skills[skill] + 0.04, 0.95)
    return found_skills


# ── Strategy 3: Context-Based Confidence ──────────────────────────────────────

_STRONG_CONTEXT_PATTERNS = [
    r"(?:expert|proficient|experienced|advanced|strong)\s+(?:in\s+)?{skill}",
    r"{skill}\s+(?:developer|engineer|architect|specialist)",
    r"built\s+(?:with|using|in)\s+{skill}",
    r"developed\s+(?:with|using|in)\s+{skill}",
    r"years?\s+(?:of\s+)?(?:experience\s+(?:with|in)\s+)?{skill}",
]

_WEAK_CONTEXT_PATTERNS = [
    r"familiar\s+(?:with\s+)?{skill}",
    r"basic\s+knowledge\s+of\s+{skill}",
    r"learning\s+{skill}",
    r"exposure\s+to\s+{skill}",
]


def _context_adjustment(text: str, found_skills: dict[str, float]) -> dict[str, float]:
    """Adjust confidence up/down based on surrounding context words."""
    text_lower = text.lower()
    for skill in list(found_skills.keys()):
        skill_lower = skill.lower()

        # Strong positive context → boost
        for ctx_pattern in _STRONG_CONTEXT_PATTERNS:
            pattern = ctx_pattern.replace("{skill}", re.escape(skill_lower))
            if re.search(pattern, text_lower):
                found_skills[skill] = min(found_skills[skill] + 0.06, 0.97)
                break

        # Weak/uncertain context → reduce
        for ctx_pattern in _WEAK_CONTEXT_PATTERNS:
            pattern = ctx_pattern.replace("{skill}", re.escape(skill_lower))
            if re.search(pattern, text_lower):
                found_skills[skill] = max(found_skills[skill] - 0.15, 0.30)
                break

    return found_skills


# ── Strategy 4: spaCy NER Noun-Chunk Extraction ───────────────────────────────

def _spacy_extract(text: str) -> dict[str, float]:
    """Use spaCy noun chunks to catch skills not in the dictionary."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:50_000])

        extra: dict[str, float] = {}
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            chunk_lower = chunk_text.lower()

            # Check if it's in our skill set (already handled by dict)
            if chunk_lower in ALL_SKILLS:
                canonical = SKILL_CANONICAL.get(chunk_lower, chunk_text)
                extra[canonical] = max(extra.get(canonical, 0), 0.75)

        # Also check named entities for ORG (often tech company tools)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT") and ent.text.lower() in ALL_SKILLS:
                canonical = SKILL_CANONICAL.get(ent.text.lower(), ent.text)
                extra[canonical] = max(extra.get(canonical, 0), 0.78)

        return extra

    except Exception as e:
        logger.warning(f"spaCy extraction failed: {e}")
        return {}


# ── Main Extraction Function ───────────────────────────────────────────────────

def extract_skills(text: str) -> list[dict]:
    """
    Full skill extraction pipeline.

    Args:
        text: Cleaned portfolio/CV text

    Returns:
        List of dicts: [{"name": "React", "confidence": 0.93, ...}, ...]
        Sorted by confidence descending.
    """
    if not text or len(text) < 50:
        return []

    # Run all strategies
    found = _dictionary_match(text)
    spacy_found = _spacy_extract(text)

    # Merge: take max confidence from any strategy
    for skill, conf in spacy_found.items():
        if skill not in found:
            found[skill] = conf

    # Apply frequency boost and context adjustment
    found = _frequency_boost(text, found)
    found = _context_adjustment(text, found)

    # Build output
    skills = []
    skill_names = []
    for skill_name, confidence in found.items():
        category = _get_skill_category(skill_name)
        skills.append({
            "name": skill_name,
            "confidence": round(confidence, 3),
            "category": category,
            "verified": False,
            "source": "nlp",
        })

    # Sort by confidence descending
    skills.sort(key=lambda x: x["confidence"], reverse=True)

    logger.debug(f"Extracted {len(skills)} skills from text")
    return skills


def _get_skill_category(skill_name: str) -> str:
    """Return the category for a skill name."""
    skill_lower = skill_name.lower()
    for category, skills in SKILL_DICTIONARY.items():
        for s in skills:
            if s.lower() == skill_lower:
                return category
    return "other"


# ── Skill Overlap Calculation ─────────────────────────────────────────────────

def calculate_skill_overlap(
    freelancer_skills: list[str],
    required_skills: list[str],
) -> tuple[list[str], list[str], float]:
    """
    Compare freelancer skills vs project requirements.

    Returns:
        (matching_skills, missing_skills, overlap_score 0.0–1.0)
    """
    freelancer_set = {s.lower() for s in freelancer_skills}
    required_set = {s.lower() for s in required_skills}

    # Also check aliases
    expanded_freelancer = set(freelancer_set)
    for skill in freelancer_set:
        if skill in SKILL_ALIASES:
            expanded_freelancer.add(SKILL_ALIASES[skill].lower())

    matching = [s for s in required_skills if s.lower() in expanded_freelancer]
    missing = [s for s in required_skills if s.lower() not in expanded_freelancer]

    overlap_score = len(matching) / len(required_set) if required_set else 0.0

    return matching, missing, round(overlap_score, 3)