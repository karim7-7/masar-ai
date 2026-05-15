"""
tests/test_api.py
──────────────────
Integration tests for all Masar AI endpoints.
Uses pytest + FastAPI TestClient (sync).

Run with:
    pytest                        # uses pytest.ini config
    pytest tests/test_api.py -v   # explicit path
    pytest -k TestRanking         # run one class only
"""

import pytest
import base64
from unittest.mock import AsyncMock, patch

# ── All heavy startup hooks are patched BEFORE main.py is imported ────────────
# Patching at module level (not inside a fixture) is the correct way to avoid
# the PytestUnraisableExceptionWarning that appears when `from main import app`
# runs inside a fixture body — the lifespan coroutines get scheduled on the
# wrong event loop at collection time.

_patches = [
    patch("app.db.database.connect_db",          new_callable=AsyncMock),
    patch("app.db.database.close_db",             new_callable=AsyncMock),
    patch("app.models.mongo_models.create_indexes", new_callable=AsyncMock),
    patch("app.services.embedding.embedding_service.get_embedding_model"),
    patch("app.services.embedding.embedding_service.load_indexes_from_mongodb",
          new_callable=AsyncMock),
]

# Start all patches before any test module code runs
for p in _patches:
    p.start()

# NOW it is safe to import — lifespan hooks will call our mocks, not real I/O
from main import app  # noqa: E402  (import not at top — intentional)
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Shared FastAPI test client for the whole test module."""
    with TestClient(app) as c:
        yield c


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_endpoint_exists(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data

    def test_root_redirect(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioAnalysis:
    BASE_URL = "/analyze-portfolio"

    def test_missing_all_inputs_returns_422(self, client):
        """Must provide at least one input source."""
        response = client.post(self.BASE_URL, json={"freelancer_id": "F001"})
        assert response.status_code == 422

    @patch("app.api.endpoints.portfolio.analyze_portfolio", new_callable=AsyncMock)
    def test_text_input_accepted(self, mock_analyze, client):
        mock_analyze.return_value = _mock_analysis_response("F001")
        response = client.post(self.BASE_URL, json={
            "freelancer_id": "F001",
            "portfolio_text": "Senior React developer with 5 years experience. Built Docker microservices.",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["freelancer_id"] == "F001"
        assert "skills" in data
        assert "portfolio_score" in data

    @patch("app.api.endpoints.portfolio.analyze_portfolio", new_callable=AsyncMock)
    def test_github_url_accepted(self, mock_analyze, client):
        mock_analyze.return_value = _mock_analysis_response("F002")
        response = client.post(self.BASE_URL, json={
            "freelancer_id": "F002",
            "github_url": "https://github.com/testuser",
        })
        assert response.status_code == 200

    @patch("app.api.endpoints.portfolio.analyze_portfolio", new_callable=AsyncMock)
    def test_pdf_base64_accepted(self, mock_analyze, client):
        mock_analyze.return_value = _mock_analysis_response("F003")
        fake_pdf_b64 = base64.b64encode(b"%PDF-1.4 test content").decode()
        response = client.post(self.BASE_URL, json={
            "freelancer_id": "F003",
            "pdf_base64": fake_pdf_b64,
        })
        assert response.status_code == 200

    @patch("app.api.endpoints.portfolio.analyze_portfolio", new_callable=AsyncMock)
    def test_response_schema_valid(self, mock_analyze, client):
        mock_analyze.return_value = _mock_analysis_response("F004")
        response = client.post(self.BASE_URL, json={
            "freelancer_id": "F004",
            "portfolio_text": "Python developer",
        })
        data = response.json()
        required_fields = [
            "freelancer_id", "skills", "verified_skills",
            "experience_level", "years_of_experience",
            "project_complexity", "portfolio_score",
            "technical_depth_score", "project_realism_score",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    @patch("app.api.endpoints.portfolio.analyze_portfolio", new_callable=AsyncMock)
    def test_experience_level_valid_enum(self, mock_analyze, client):
        mock_analyze.return_value = _mock_analysis_response("F005")
        response = client.post(self.BASE_URL, json={
            "freelancer_id": "F005",
            "portfolio_text": "Developer",
        })
        data = response.json()
        assert data["experience_level"] in ["Beginner", "Intermediate", "Expert"]


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbeddings:
    BASE_URL = "/generate-embedding"

    @patch("app.api.endpoints.matching.store_embedding", new_callable=AsyncMock)
    def test_freelancer_embedding(self, mock_store, client):
        import numpy as np
        mock_store.return_value = (np.zeros(384, dtype="float32"), 0)
        response = client.post(self.BASE_URL, json={
            "entity_id": "F001",
            "entity_type": "freelancer",
            "text": "React developer with 3 years experience building SaaS products",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["entity_id"] == "F001"
        assert data["embedding_dimension"] == 384

    @patch("app.api.endpoints.matching.store_embedding", new_callable=AsyncMock)
    def test_project_embedding(self, mock_store, client):
        import numpy as np
        mock_store.return_value = (np.zeros(384, dtype="float32"), 1)
        response = client.post(self.BASE_URL, json={
            "entity_id": "P001",
            "entity_type": "project",
            "text": "Looking for React + Node.js developer for e-commerce platform",
        })
        assert response.status_code == 201

    def test_invalid_entity_type(self, client):
        response = client.post(self.BASE_URL, json={
            "entity_id": "X001",
            "entity_type": "invalid_type",
            "text": "Some text",
        })
        assert response.status_code == 422

    def test_empty_text_rejected(self, client):
        response = client.post(self.BASE_URL, json={
            "entity_id": "F001",
            "entity_type": "freelancer",
            "text": "short",   # < 10 chars
        })
        assert response.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# MATCHING
# ══════════════════════════════════════════════════════════════════════════════

class TestMatching:
    BASE_URL = "/match-project"

    @patch("app.api.endpoints.matching.match_project_to_freelancers", new_callable=AsyncMock)
    def test_basic_match(self, mock_match, client):
        mock_match.return_value = _mock_match_response("P001")
        response = client.post(self.BASE_URL, json={
            "project_id": "P001",
            "title": "E-commerce Platform",
            "description": "Build a full-stack e-commerce platform with React and Node.js",
            "required_skills": ["React", "Node.js", "MongoDB"],
            "complexity": "Advanced",
            "experience_required": "Intermediate",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == "P001"
        assert "recommended_freelancers" in data

    @patch("app.api.endpoints.matching.match_project_to_freelancers", new_callable=AsyncMock)
    def test_match_result_has_required_fields(self, mock_match, client):
        mock_match.return_value = _mock_match_response("P002")
        response = client.post(self.BASE_URL, json={
            "project_id": "P002",
            "title": "Test",
            "description": "Test project",
        })
        data = response.json()
        if data["recommended_freelancers"]:
            freelancer = data["recommended_freelancers"][0]
            assert "freelancer_id" in freelancer
            assert "match_score" in freelancer
            assert "matching_skills" in freelancer
            assert "missing_skills" in freelancer
            assert "reason" in freelancer

    def test_invalid_top_k(self, client):
        response = client.post(f"{self.BASE_URL}?top_k=100", json={
            "project_id": "P003",
            "title": "Test",
            "description": "Test",
        })
        assert response.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# RANKING
# ══════════════════════════════════════════════════════════════════════════════

class TestRanking:
    BASE_URL = "/rank-freelancer"

    @patch("app.api.endpoints.ranking.rank_freelancer", new_callable=AsyncMock)
    def test_basic_ranking(self, mock_rank, client):
        mock_rank.return_value = _mock_ranking_response("F001")
        response = client.post(self.BASE_URL, json={
            "freelancer_id": "F001",
            "skill_relevance_raw": 85,
            "portfolio_score": 78,
            "avg_client_rating": 4.5,
            "response_time_hours": 2.0,
            "completed_projects": 12,
            "experience_level": "Intermediate",
        })
        assert response.status_code == 200
        data = response.json()
        assert "final_score" in data
        assert 0 <= data["final_score"] <= 100

    @patch("app.api.endpoints.ranking.rank_freelancer", new_callable=AsyncMock)
    def test_spam_profile(self, mock_rank, client):
        mock_rank.return_value = _mock_ranking_response("F002", is_spam=True, score=22.0)
        response = client.post(self.BASE_URL, json={
            "freelancer_id": "F002",
            "skill_relevance_raw": 99,
            "portfolio_score": 99,
            "avg_client_rating": 5.0,
            "response_time_hours": 0.1,
            "completed_projects": 0,
            "experience_level": "Expert",
            "is_spam_suspected": True,
        })
        assert response.status_code == 200

    @patch("app.api.endpoints.ranking.rank_freelancer", new_callable=AsyncMock)
    def test_ranking_reasons_present(self, mock_rank, client):
        mock_rank.return_value = _mock_ranking_response("F003")
        response = client.post(self.BASE_URL, json={
            "freelancer_id": "F003",
            "skill_relevance_raw": 70,
            "portfolio_score": 75,
            "avg_client_rating": 4.2,
            "response_time_hours": 5,
            "completed_projects": 8,
            "experience_level": "Intermediate",
        })
        data = response.json()
        assert isinstance(data["ranking_reasons"], list)
        assert len(data["ranking_reasons"]) > 0

    def test_invalid_rating_range(self, client):
        response = client.post(self.BASE_URL, json={
            "freelancer_id": "F004",
            "skill_relevance_raw": 80,
            "portfolio_score": 80,
            "avg_client_rating": 10,   # Invalid: max is 5
            "response_time_hours": 3,
            "completed_projects": 5,
            "experience_level": "Intermediate",
        })
        assert response.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — NLP Services
# ══════════════════════════════════════════════════════════════════════════════

class TestNLPServices:

    def test_skill_extraction_basic(self):
        from app.services.nlp.skill_extractor import extract_skills
        text = "I am a React developer with experience in Node.js, MongoDB, and Docker."
        skills = extract_skills(text)
        skill_names = [s["name"] for s in skills]
        assert "React" in skill_names
        assert "Node.js" in skill_names
        assert "MongoDB" in skill_names

    def test_skill_extraction_empty_text(self):
        from app.services.nlp.skill_extractor import extract_skills
        assert extract_skills("") == []
        assert extract_skills("Hi there") == []

    def test_skill_overlap_calculation(self):
        from app.services.nlp.skill_extractor import calculate_skill_overlap
        matching, missing, score = calculate_skill_overlap(
            freelancer_skills=["React", "Python", "Docker"],
            required_skills=["React", "Docker", "Kubernetes"],
        )
        assert "React" in matching
        assert "Docker" in matching
        assert "Kubernetes" in missing
        assert abs(score - 0.667) < 0.01

    def test_years_extraction(self):
        from app.services.nlp.preprocessor import extract_years_of_experience
        texts = [
            ("5 years of experience in software development", 5),
            ("over 3 years building web applications", 3),
            ("I have 7+ years working with Python", 7),
            ("No experience stated", 0),
        ]
        for text, expected in texts:
            result = extract_years_of_experience(text)
            assert result == expected, f"Expected {expected}, got {result} for: {text}"

    def test_clean_text_removes_urls(self):
        from app.services.nlp.preprocessor import clean_text
        text = "Check my portfolio at https://mysite.com and contact me at test@email.com"
        cleaned = clean_text(text)
        assert "https://" not in cleaned
        assert "@email.com" not in cleaned

    def test_ranking_normalize_rating(self):
        from app.services.ranking.ranking_engine import normalize_rating
        assert normalize_rating(5.0) == 100.0
        assert normalize_rating(0.0) == 0.0
        assert normalize_rating(2.5) == 50.0

    def test_ranking_response_time_normalization(self):
        from app.services.ranking.ranking_engine import normalize_response_time
        assert normalize_response_time(0.5) == 100.0
        assert normalize_response_time(2.0) == 90.0
        assert normalize_response_time(200) == 5.0

    def test_spam_detection(self):
        from app.services.ranking.ranking_engine import detect_spam_signals
        from app.schemas.schemas import RankingInput

        spam_input = RankingInput(
            freelancer_id="SPAM001",
            skill_relevance_raw=99,
            portfolio_score=99,
            avg_client_rating=5.0,
            response_time_hours=0.1,
            completed_projects=0,
            experience_level="Expert",
            is_spam_suspected=True,
        )
        is_spam, penalty = detect_spam_signals(spam_input)
        assert is_spam is True
        assert penalty < 0.5

    def test_beginner_boost_applied(self):
        from app.services.ranking.ranking_engine import apply_beginner_fairness
        from app.schemas.schemas import RankingInput

        data = RankingInput(
            freelancer_id="B001",
            skill_relevance_raw=70,
            portfolio_score=75,
            avg_client_rating=4.0,
            response_time_hours=5,
            completed_projects=2,
            experience_level="Beginner",
        )
        boosted, was_boosted = apply_beginner_fairness(60.0, data)
        assert was_boosted is True
        assert boosted == 70.0

    def test_beginner_boost_not_applied_low_portfolio(self):
        from app.services.ranking.ranking_engine import apply_beginner_fairness
        from app.schemas.schemas import RankingInput

        data = RankingInput(
            freelancer_id="B002",
            skill_relevance_raw=40,
            portfolio_score=30,       # Too low for boost
            avg_client_rating=3.0,
            response_time_hours=10,
            completed_projects=0,
            experience_level="Beginner",
        )
        score, was_boosted = apply_beginner_fairness(50.0, data)
        assert was_boosted is False
        assert score == 50.0


# ══════════════════════════════════════════════════════════════════════════════
# MOCK FACTORIES
# ══════════════════════════════════════════════════════════════════════════════

def _mock_analysis_response(freelancer_id: str):
    from app.schemas.schemas import PortfolioAnalysisResponse, SkillScore
    return PortfolioAnalysisResponse(
        freelancer_id=freelancer_id,
        skills=[
            SkillScore(name="React", confidence=0.93),
            SkillScore(name="Node.js", confidence=0.88),
        ],
        verified_skills=["React", "Node.js"],
        experience_level="Intermediate",
        years_of_experience=3,
        project_complexity="Medium",
        portfolio_score=78.5,
        technical_depth_score=75.0,
        project_realism_score=80.0,
        analysis_source=["pdf"],
        raw_text_length=2500,
    )


def _mock_match_response(project_id: str):
    from app.schemas.schemas import MatchProjectResponse, FreelancerMatchResult
    return MatchProjectResponse(
        project_id=project_id,
        total_candidates=3,
        recommended_freelancers=[
            FreelancerMatchResult(
                freelancer_id="F001",
                match_score=89.5,
                matching_skills=["React", "Node.js"],
                missing_skills=["Docker"],
                semantic_similarity=0.82,
                skill_overlap_score=0.75,
                reason="Strong React experience",
            )
        ],
    )


def _mock_ranking_response(freelancer_id: str, is_spam=False, score=82.5):
    from app.schemas.schemas import RankingResponse
    return RankingResponse(
        freelancer_id=freelancer_id,
        final_score=score,
        component_scores={
            "skill_relevance": 85.0,
            "portfolio_quality": 78.0,
            "client_rating": 90.0,
            "response_speed": 90.0,
        },
        ranking_reasons=["Strong verified React experience", "Good client satisfaction"],
        is_spam=is_spam,
        beginner_boost_applied=False,
    )