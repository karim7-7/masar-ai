"""
app/services/nlp/github_analyzer.py
─────────────────────────────────────
Analyze a GitHub profile or repository URL to:
  • Extract languages used (verified from actual code)
  • Count repositories, stars, contributions
  • Detect tech stack from repo topics / READMEs
  • Generate a "verified skills" set from real code

Uses PyGithub. Falls back gracefully if rate-limited or token missing.
"""

import re
import base64

from loguru import logger
from app.core.config import settings


def _get_github_client():
    """Return authenticated GitHub client or None."""

    try:
        from github import Github

        token = settings.github_token

        if token:
            return Github(token)

        logger.warning(
            "No GITHUB_TOKEN set — using unauthenticated GitHub API (rate limited)"
        )

        return Github()

    except ImportError:

        logger.warning(
            "PyGithub not installed — skipping GitHub analysis"
        )

        return None


def _parse_github_url(url: str) -> tuple[str | None, str | None]:
    """
    Parse a GitHub URL and return (username, repo_name).
    repo_name is None for profile URLs.
    """

    # Profile URL
    profile_match = re.match(
        r"https?://github\.com/([^/]+)/?$",
        url.strip()
    )

    if profile_match:
        return profile_match.group(1), None

    # Repo URL
    repo_match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
        url.strip()
    )

    if repo_match:
        return repo_match.group(1), repo_match.group(2)

    return None, None


def analyze_github_profile(username: str, gh) -> dict:
    """Analyze a user's public GitHub profile."""

    try:

        user = gh.get_user(username)

        repos = list(user.get_repos())

        languages: dict[str, int] = {}
        topics: list[str] = []

        total_stars = 0

        repo_names = []

        for repo in repos[:5]:

            # Skip forks
            if repo.fork:
                continue

            repo_names.append(repo.name)

            total_stars += repo.stargazers_count

            # Language breakdown
            try:

                repo_langs = repo.get_languages()

                for lang, bytes_count in repo_langs.items():

                    languages[lang] = (
                        languages.get(lang, 0)
                        + bytes_count
                    )

            except Exception:

                if repo.language:

                    languages[repo.language] = (
                        languages.get(repo.language, 0)
                        + 1
                    )

            # Topics
            try:
                topics.extend(repo.get_topics())

            except Exception:
                pass

        # Sort languages
        sorted_langs = sorted(
            languages.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_languages = [
            lang for lang, _ in sorted_langs[:10]
        ]

        return {

            "username": username,

            "public_repos": user.public_repos,

            "followers": user.followers,

            "total_stars": total_stars,

            "repo_names": repo_names,

            "top_languages": top_languages,

            "topics": list(set(topics)),

            "account_age_years": _calculate_account_age(
                user.created_at
            ),

            "success": True,
        }

    except Exception as e:

        logger.warning(
            f"GitHub profile analysis failed for {username}: {e}"
        )

        return {
            "success": False,
            "error": str(e)
        }


def analyze_github_repo(username: str, repo_name: str, gh) -> dict:
    """Analyze a single GitHub repository."""

    try:

        repo = gh.get_repo(f"{username}/{repo_name}")

        languages = {}

        try:
            languages = repo.get_languages()

        except Exception:

            if repo.language:
                languages = {repo.language: 100}

        readme_text = ""

        try:

            readme = repo.get_readme()

            readme_text = base64.b64decode(
                readme.content
            ).decode(
                "utf-8",
                errors="ignore"
            )

        except Exception:
            pass

        return {

            "repo_name": repo.name,

            "description": repo.description,

            "stars": repo.stargazers_count,

            "forks": repo.forks_count,

            "languages": list(languages.keys()),

            "topics": repo.get_topics(),

            "readme_text": readme_text[:3000],

            "html_url": repo.html_url,

            "is_fork": repo.fork,

            "success": True,
        }

    except Exception as e:

        logger.warning(
            f"GitHub repo analysis failed "
            f"{username}/{repo_name}: {e}"
        )

        return {
            "success": False,
            "error": str(e)
        }


def _calculate_account_age(created_at) -> float:
    """Return account age in years."""

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    if created_at.tzinfo is None:
        created_at = created_at.replace(
            tzinfo=timezone.utc
        )

    delta = now - created_at

    return round(delta.days / 365.25, 1)


def analyze_github_url(url: str) -> dict:
    """
    Main entry point:
    takes a GitHub URL and returns analysis results.
    """

    gh = _get_github_client()

    if gh is None:

        return {
            "success": False,
            "error": "GitHub client unavailable"
        }

    username, repo_name = _parse_github_url(url)

    if not username:

        return {
            "success": False,
            "error": f"Cannot parse GitHub URL: {url}"
        }

    # Single repo
    if repo_name:

        result = analyze_github_repo(
            username=username,
            repo_name=repo_name,
            gh=gh,
        )

        if result.get("success"):

            result["verified_skills"] = (
                result.get("languages", [])
            )

    # Profile
    else:

        result = analyze_github_profile(
            username=username,
            gh=gh,
        )

        if result.get("success"):

            result["verified_skills"] = (
                result.get("top_languages", [])
            )

    return result