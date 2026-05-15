"""
scripts/demo_requests.py
─────────────────────────
Live demo script — sends real HTTP requests to the running Masar AI service.
Showcases all 5 endpoints with realistic data.

Usage:
    python scripts/demo_requests.py
    python scripts/demo_requests.py --host http://localhost:8000
"""

import argparse
import base64
import json
import sys
import time
import httpx

BASE_URL = "http://localhost:8000"

# ── Colors for terminal output ─────────────────────────────────────────────────
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(title: str):
    print(f"\n{BOLD}{BLUE}{'═'*60}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'═'*60}{RESET}")


def print_request(method: str, url: str, body: dict):
    print(f"\n{YELLOW}▶ REQUEST:{RESET} {BOLD}{method} {url}{RESET}")
    print(f"{YELLOW}Body:{RESET}")
    print(json.dumps(body, indent=2))


def print_response(response: httpx.Response):
    status_color = GREEN if response.status_code < 400 else RED
    print(f"\n{status_color}◀ RESPONSE: {response.status_code}{RESET}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(response.text)
    print()


def demo_health(client: httpx.Client):
    print_header("1. HEALTH CHECK  →  GET /health")
    response = client.get(f"{BASE_URL}/health")
    print_response(response)
    return response.status_code == 200


def demo_portfolio_analysis(client: httpx.Client) -> str:
    print_header("2. PORTFOLIO ANALYSIS  →  POST /analyze-portfolio")

    # Realistic freelancer portfolio text
    portfolio_text = """
    Senior Full-Stack Developer | 5 Years of Experience

    I am an experienced software engineer specializing in React, Node.js, and cloud
    architecture. Over the past 5 years, I have designed and deployed scalable
    microservices serving over 500,000 users.

    SKILLS:
    Frontend: React, Next.js, TypeScript, Tailwind CSS, Redux
    Backend: Node.js, Express, FastAPI, Python, REST APIs, GraphQL
    Databases: MongoDB, PostgreSQL, Redis
    DevOps: Docker, Kubernetes, AWS EC2, GitHub Actions, CI/CD pipelines
    AI/ML: TensorFlow, scikit-learn, Pandas, NumPy

    KEY PROJECTS:
    1. E-Commerce Platform (Advanced)
       Built a full-stack e-commerce platform with React + Node.js.
       Deployed on AWS with Docker and Kubernetes. Handles 10,000+ daily transactions.
       Integrated Stripe payment gateway and Redis caching layer.

    2. Real-Time Analytics Dashboard (Advanced)
       Designed and implemented a real-time data pipeline using WebSockets and
       MongoDB aggregation. Dashboard serves 200+ enterprise clients.
       Used Chart.js and D3.js for data visualization.

    3. AI Content Moderation System (Expert-level)
       Built an NLP-based content moderation pipeline using Python and TensorFlow.
       Reduced manual review time by 80%. Deployed on Google Cloud with CI/CD.

    GitHub: https://github.com/masar-demo-user
    """

    body = {
        "freelancer_id": "F001",
        "portfolio_text": portfolio_text,
        "github_url": "https://github.com/torvalds",  # Public profile for demo
    }

    print_request("POST", "/analyze-portfolio", {"freelancer_id": "F001", "portfolio_text": "..."})
    response = client.post(f"{BASE_URL}/analyze-portfolio", json=body, timeout=60)
    print_response(response)
    return "F001"


def demo_generate_embeddings(client: httpx.Client):
    print_header("3. GENERATE EMBEDDINGS  →  POST /generate-embedding")

    # Freelancer embedding
    freelancer_body = {
        "entity_id": "F001",
        "entity_type": "freelancer",
        "text": (
            "Senior React and Node.js developer with 5 years experience. "
            "Skills: React, Next.js, Node.js, MongoDB, Docker, Kubernetes, AWS, Python. "
            "Experience level: Expert. Built scalable microservices for 500k users."
        ),
    }

    print_request("POST", "/generate-embedding", freelancer_body)
    response = client.post(f"{BASE_URL}/generate-embedding", json=freelancer_body, timeout=30)
    print_response(response)

    # Add a second freelancer
    time.sleep(0.5)
    freelancer2_body = {
        "entity_id": "F002",
        "entity_type": "freelancer",
        "text": (
            "Python backend developer with 3 years experience. "
            "Skills: Python, Django, PostgreSQL, Redis, Docker. "
            "Experience level: Intermediate. Built REST APIs for fintech startup."
        ),
    }
    print_request("POST", "/generate-embedding", freelancer2_body)
    response2 = client.post(f"{BASE_URL}/generate-embedding", json=freelancer2_body, timeout=30)
    print_response(response2)

    # Project embedding
    time.sleep(0.5)
    project_body = {
        "entity_id": "P001",
        "entity_type": "project",
        "text": (
            "E-commerce platform development. "
            "Required skills: React, Node.js, MongoDB, Docker. "
            "Full-stack web application with payment integration. "
            "Complexity: Advanced. Experience required: Intermediate."
        ),
    }
    print_request("POST", "/generate-embedding", project_body)
    response3 = client.post(f"{BASE_URL}/generate-embedding", json=project_body, timeout=30)
    print_response(response3)


def demo_match_project(client: httpx.Client):
    print_header("4. SMART MATCHING  →  POST /match-project")

    body = {
        "project_id": "P001",
        "title": "Full-Stack E-Commerce Platform",
        "description": (
            "We need an experienced full-stack developer to build a scalable "
            "e-commerce platform. The platform must handle real-time inventory, "
            "payment processing, and an admin dashboard. Must be deployed on AWS."
        ),
        "required_skills": ["React", "Node.js", "MongoDB", "Docker", "AWS"],
        "budget_range": "$5,000 - $10,000",
        "complexity": "Advanced",
        "experience_required": "Intermediate",
    }

    print_request("POST", "/match-project", body)
    response = client.post(f"{BASE_URL}/match-project?top_k=5", json=body, timeout=30)
    print_response(response)


def demo_rank_freelancer(client: httpx.Client):
    print_header("5. FREELANCER RANKING  →  POST /rank-freelancer")

    # Expert freelancer
    body_expert = {
        "freelancer_id": "F001",
        "skill_relevance_raw": 88,
        "portfolio_score": 85,
        "avg_client_rating": 4.8,
        "response_time_hours": 1.5,
        "completed_projects": 24,
        "experience_level": "Expert",
        "is_spam_suspected": False,
    }
    print_request("POST", "/rank-freelancer", body_expert)
    r1 = client.post(f"{BASE_URL}/rank-freelancer", json=body_expert, timeout=20)
    print_response(r1)

    time.sleep(0.3)

    # Beginner with strong portfolio (tests boost)
    body_beginner = {
        "freelancer_id": "F003",
        "skill_relevance_raw": 72,
        "portfolio_score": 78,
        "avg_client_rating": 0.0,
        "response_time_hours": 0.8,
        "completed_projects": 0,
        "experience_level": "Beginner",
        "is_spam_suspected": False,
    }
    print(f"\n{YELLOW}Testing BEGINNER FAIRNESS BOOST...{RESET}")
    print_request("POST", "/rank-freelancer", body_beginner)
    r2 = client.post(f"{BASE_URL}/rank-freelancer", json=body_beginner, timeout=20)
    print_response(r2)

    time.sleep(0.3)

    # Spam profile (tests penalization)
    body_spam = {
        "freelancer_id": "SPAM001",
        "skill_relevance_raw": 99,
        "portfolio_score": 99,
        "avg_client_rating": 5.0,
        "response_time_hours": 0.01,
        "completed_projects": 0,
        "experience_level": "Expert",
        "is_spam_suspected": True,
    }
    print(f"\n{RED}Testing SPAM DETECTION...{RESET}")
    print_request("POST", "/rank-freelancer", body_spam)
    r3 = client.post(f"{BASE_URL}/rank-freelancer", json=body_spam, timeout=20)
    print_response(r3)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Masar AI Demo Script")
    parser.add_argument("--host", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.host.rstrip("/")

    print(f"\n{BOLD}🚀 Masar AI — Live Demo{RESET}")
    print(f"Target: {BOLD}{BASE_URL}{RESET}")
    print(f"{'─'*60}")

    with httpx.Client(timeout=120) as client:
        # 1. Health check
        healthy = demo_health(client)
        if not healthy:
            print(f"\n{RED}❌ Service is not healthy. Make sure it's running.{RESET}")
            print(f"   docker compose up   OR   uvicorn main:app --reload")
            sys.exit(1)

        # 2. Portfolio analysis
        demo_portfolio_analysis(client)

        # 3. Generate embeddings (needed before matching)
        demo_generate_embeddings(client)

        # 4. Match project
        demo_match_project(client)

        # 5. Rank freelancer
        demo_rank_freelancer(client)

    print(f"\n{GREEN}{BOLD}✅ Demo complete!{RESET}")
    print(f"View API docs at: {BASE_URL}/docs\n")


if __name__ == "__main__":
    main()