"""
app/services/embedding/embedding_service.py
─────────────────────────────────────────────
Manages:
  • Generating text embeddings with sentence-transformers (all-MiniLM-L6-v2)
  • Storing / searching embeddings in FAISS
  • Persisting embedding metadata to MongoDB
"""

import os
import numpy as np
from loguru import logger
from app.core.config import settings


# ── Singleton state ────────────────────────────────────────────────────────────

_model = None
_freelancer_index = None   # FAISS index for freelancers
_project_index = None      # FAISS index for projects

# Map: FAISS internal integer ID → entity_id string
_freelancer_id_map: dict[int, str] = {}
_project_id_map: dict[int, str] = {}

# Next available FAISS integer IDs
_next_freelancer_id = 0
_next_project_id = 0


# ── Model Loading ─────────────────────────────────────────────────────────────

def get_embedding_model():
    """Load sentence-transformer model (singleton, thread-safe enough for async)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded.")
    return _model


# ── FAISS Index Management ────────────────────────────────────────────────────

def _get_faiss_index(entity_type: str):
    """Get or create FAISS IndexFlatIP (inner product = cosine after normalization)."""
    global _freelancer_index, _project_index
    import faiss

    dim = settings.embedding_dimension

    if entity_type == "freelancer":
        if _freelancer_index is None:
            _freelancer_index = _load_or_create_index(
                settings.faiss_freelancer_index, dim
            )
        return _freelancer_index
    else:
        if _project_index is None:
            _project_index = _load_or_create_index(
                settings.faiss_project_index, dim
            )
        return _project_index


def _load_or_create_index(filename: str, dim: int):
    """Load FAISS index from disk or create a new one."""
    import faiss
    os.makedirs(settings.faiss_index_path, exist_ok=True)
    path = os.path.join(settings.faiss_index_path, filename)

    if os.path.exists(path):
        index = faiss.read_index(path)
        logger.info(f"FAISS index loaded from {path} ({index.ntotal} vectors)")
    else:
        # IndexFlatIP: exact inner product search (use normalized vectors → cosine sim)
        index = faiss.IndexFlatIP(dim)
        logger.info(f"Created new FAISS index (dim={dim}): {path}")
    return index


def _save_index(entity_type: str) -> None:
    """Persist FAISS index to disk."""
    import faiss
    filename = (
        settings.faiss_freelancer_index
        if entity_type == "freelancer"
        else settings.faiss_project_index
    )
    path = os.path.join(settings.faiss_index_path, filename)
    index = _get_faiss_index(entity_type)
    faiss.write_index(index, path)
    logger.debug(f"FAISS index saved: {path}")


# ── Core Embedding Functions ───────────────────────────────────────────────────

def generate_embedding(text: str) -> np.ndarray:
    """
    Generate L2-normalized embedding vector for the given text.
    Normalization enables cosine similarity via inner product in FAISS.
    """
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return embedding.astype(np.float32)


def build_freelancer_text(profile: dict) -> str:
    """
    Construct a rich text representation of a freelancer profile
    for embedding generation. More context = better semantic matching.
    """
    parts = []

    if profile.get("bio"):
        parts.append(profile["bio"])

    if profile.get("skills"):
        skills = profile["skills"]
        # Handle both list-of-strings and list-of-dicts
        if skills and isinstance(skills[0], dict):
            skill_names = [s["name"] for s in skills]
        else:
            skill_names = skills
        parts.append(f"Skills: {', '.join(skill_names)}")

    if profile.get("experience_level"):
        parts.append(f"Experience level: {profile['experience_level']}")

    if profile.get("years_of_experience"):
        parts.append(f"Years of experience: {profile['years_of_experience']}")

    if profile.get("portfolio_description"):
        parts.append(profile["portfolio_description"])

    if profile.get("verified_skills"):
        parts.append(f"Verified skills: {', '.join(profile['verified_skills'])}")

    return " | ".join(parts)


def build_project_text(project: dict) -> str:
    """Build text representation of a project for embedding."""
    parts = []

    if project.get("title"):
        parts.append(project["title"])

    if project.get("description"):
        parts.append(project["description"])

    if project.get("required_skills"):
        parts.append(f"Required skills: {', '.join(project['required_skills'])}")

    if project.get("complexity"):
        parts.append(f"Complexity: {project['complexity']}")

    if project.get("experience_required"):
        parts.append(f"Experience required: {project['experience_required']}")

    return " | ".join(parts)


# ── Store Embedding ────────────────────────────────────────────────────────────

async def store_embedding(
    entity_id: str,
    entity_type: str,
    text: str,
    db=None,
) -> tuple[np.ndarray, int]:
    """
    Generate embedding and add to FAISS + MongoDB.

    Returns:
        (embedding_vector, faiss_internal_id)
    """
    global _next_freelancer_id, _next_project_id

    embedding = generate_embedding(text)
    index = _get_faiss_index(entity_type)

    # Determine next FAISS ID
    if entity_type == "freelancer":
        faiss_id = _next_freelancer_id
        _freelancer_id_map[faiss_id] = entity_id
        _next_freelancer_id += 1
    else:
        faiss_id = _next_project_id
        _project_id_map[faiss_id] = entity_id
        _next_project_id += 1

    # Add to FAISS
    index.add(embedding.reshape(1, -1))
    _save_index(entity_type)

    # Persist to MongoDB
    if db is not None:
        from app.models.mongo_models import embedding_doc
        doc = embedding_doc(
            entity_id=entity_id,
            entity_type=entity_type,
            embedding=embedding.tolist(),
            faiss_index_id=faiss_id,
            model_name=settings.embedding_model,
        )
        await db["embeddings"].replace_one(
            {"entity_id": entity_id, "entity_type": entity_type},
            doc,
            upsert=True,
        )

    logger.info(f"Stored embedding for {entity_type}:{entity_id} (FAISS id={faiss_id})")
    return embedding, faiss_id


# ── Search Similar ─────────────────────────────────────────────────────────────

async def search_similar(
    query_text: str,
    entity_type: str,
    top_k: int = 10,
    db=None,
) -> list[dict]:
    """
    Find top-k most similar entities using FAISS nearest-neighbor search.

    Returns:
        List of {entity_id, similarity_score} sorted by score descending.
    """
    query_embedding = generate_embedding(query_text)
    index = _get_faiss_index(entity_type)

    if index.ntotal == 0:
        logger.warning(f"FAISS {entity_type} index is empty")
        return []

    actual_k = min(top_k, index.ntotal)
    scores, faiss_ids = index.search(query_embedding.reshape(1, -1), actual_k)

    id_map = _freelancer_id_map if entity_type == "freelancer" else _project_id_map

    results = []
    for score, faiss_id in zip(scores[0], faiss_ids[0]):
        if faiss_id == -1:
            continue
        entity_id = id_map.get(int(faiss_id), f"unknown_{faiss_id}")
        results.append({
            "entity_id": entity_id,
            "similarity_score": float(score),  # Inner product = cosine sim (normalized)
        })

    return results


async def load_indexes_from_mongodb(db) -> None:
    """
    On startup, reload FAISS indexes from MongoDB embeddings collection.
    This ensures embeddings survive service restarts.
    """
    global _next_freelancer_id, _next_project_id, _freelancer_id_map, _project_id_map
    import faiss

    logger.info("Rebuilding FAISS indexes from MongoDB...")

    for entity_type in ["freelancer", "project"]:
        index = _get_faiss_index(entity_type)
        cursor = db["embeddings"].find({"entity_type": entity_type})
        count = 0

        async for doc in cursor:
            emb = np.array(doc["embedding"], dtype=np.float32)
            faiss_id = doc.get("faiss_index_id", count)
            index.add(emb.reshape(1, -1))

            if entity_type == "freelancer":
                _freelancer_id_map[faiss_id] = doc["entity_id"]
                _next_freelancer_id = max(_next_freelancer_id, faiss_id + 1)
            else:
                _project_id_map[faiss_id] = doc["entity_id"]
                _next_project_id = max(_next_project_id, faiss_id + 1)
            count += 1

        logger.info(f"Loaded {count} {entity_type} embeddings into FAISS")