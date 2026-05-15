"""
app/services/nlp/preprocessor.py
──────────────────────────────────
Text preprocessing pipeline for portfolio/CV text.
Steps: clean → normalize → tokenize → remove stopwords → lemmatize
"""

import re
import unicodedata
from loguru import logger

# Lazy-load heavy models on first use
_nlp = None
_stop_words = None


def _get_nlp():
    """Load spaCy model (singleton)."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded: en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found — downloading...")
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_stop_words() -> set[str]:
    """Load NLTK stopwords (singleton)."""
    global _stop_words
    if _stop_words is None:
        import nltk
        try:
            from nltk.corpus import stopwords
            _stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            _stop_words = set(stopwords.words("english"))
    return _stop_words


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Remove noise from raw extracted text:
    • Unicode normalization
    • Remove non-printable chars
    • Collapse excessive whitespace / newlines
    • Remove page numbers, headers/footers artefacts
    """
    # Normalize unicode (NFC form)
    text = unicodedata.normalize("NFC", text)

    # Remove non-printable characters (keep newlines and tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Remove URLs (we analyze GitHub separately)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove phone numbers
    text = re.sub(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]", " ", text)

    # Standardize bullets / list markers
    text = re.sub(r"^[\u2022\u2023\u25E6\u2043\-\*\•]\s*", "", text, flags=re.MULTILINE)

    # Collapse multiple spaces / tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation for bag-of-words style analysis."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Tokenization & Lemmatization ───────────────────────────────────────────────

def tokenize_and_lemmatize(text: str) -> list[str]:
    """
    Use spaCy to:
    • Tokenize the text
    • Remove stopwords and punctuation
    • Lemmatize each token
    Returns list of clean lemmas.
    """
    nlp = _get_nlp()
    stop_words = _get_stop_words()

    # spaCy processes up to 1M chars; truncate if needed
    if len(text) > 900_000:
        text = text[:900_000]

    doc = nlp(text)

    tokens = []
    for token in doc:
        if (
            not token.is_punct
            and not token.is_space
            and not token.is_stop
            and len(token.text) > 1
            and token.text.lower() not in stop_words
        ):
            tokens.append(token.lemma_.lower())

    return tokens


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences using spaCy's sentencizer."""
    nlp = _get_nlp()
    doc = nlp(text[:50_000])  # limit for speed
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]


# ── Years of Experience Extraction ────────────────────────────────────────────

def extract_years_of_experience(text: str) -> int:
    """
    Scan text for patterns like:
      "5 years of experience", "3+ years", "over 4 years", etc.
    Returns the max found value, or 0 if none found.
    """
    patterns = [
        r"(\d+)\+?\s*years?\s+of\s+(?:professional\s+)?experience",
        r"(\d+)\+?\s*years?\s+(?:working|developing|building|coding)",
        r"experience\s+(?:of\s+)?(\d+)\+?\s*years?",
        r"(\d+)\+?\s*yr[s]?\s+(?:of\s+)?(?:exp|experience)",
        r"over\s+(\d+)\s+years?",
        r"more\s+than\s+(\d+)\s+years?",
    ]
    found_years = []
    text_lower = text.lower()

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        found_years.extend(int(m) for m in matches if int(m) < 50)

    return max(found_years) if found_years else 0


# ── Full Preprocessing Pipeline ───────────────────────────────────────────────

def preprocess_portfolio_text(raw_text: str) -> dict:
    """
    Full pipeline:
    Returns dict with: cleaned_text, normalized_text, tokens, sentences, years
    """
    cleaned = clean_text(raw_text)
    normalized = normalize_text(cleaned)
    tokens = tokenize_and_lemmatize(cleaned)
    sentences = extract_sentences(cleaned)
    years = extract_years_of_experience(cleaned)

    return {
        "cleaned_text": cleaned,
        "normalized_text": normalized,
        "tokens": tokens,
        "sentences": sentences,
        "years_of_experience": years,
        "char_count": len(cleaned),
        "token_count": len(tokens),
    }