# """
# app/api/endpoints/portfolio.py
# ────────────────────────────────
# POST /analyze-portfolio
#   Accepts PDF, text, GitHub URL, portfolio URL.
#   Returns structured AI analysis of a freelancer's portfolio.
# """

# from fastapi import APIRouter, Depends, HTTPException, status
# from loguru import logger

# from app.schemas.schemas import PortfolioAnalysisRequest, PortfolioAnalysisResponse
# from app.services.nlp.portfolio_analyzer import analyze_portfolio
# from app.db.database import get_db

# router = APIRouter()


# @router.post(
#     "/analyze-portfolio",
#     response_model=PortfolioAnalysisResponse,
#     status_code=status.HTTP_200_OK,
#     summary="Analyze a freelancer portfolio",
#     description=(
#         "Accepts a base64-encoded PDF CV, portfolio text, GitHub URL, or any "
#         "combination. Returns extracted skills, experience level, portfolio "
#         "quality scores, and verified skills."
#     ),
#     tags=["Portfolio Analysis"],
# )
# async def analyze_portfolio_endpoint(
#     request: PortfolioAnalysisRequest,
#     db=Depends(get_db),
# ) -> PortfolioAnalysisResponse:
#     """
#     Full portfolio analysis pipeline.

#     **At least one** of the following must be provided:
#     - `pdf_base64` – Base64-encoded PDF bytes
#     - `portfolio_text` – Raw text description
#     - `github_url` – GitHub profile or repo URL
#     """
#     # Validate at least one input source is present
#     if not any([
#         request.pdf_base64,
#         request.portfolio_text,
#         request.github_url,
#         request.portfolio_url,
#     ]):
#         raise HTTPException(
#             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#             detail="At least one of pdf_base64, portfolio_text, github_url must be provided",
#         )

#     logger.info(f"Portfolio analysis request — freelancer_id={request.freelancer_id}")

#     try:
#         result = await analyze_portfolio(request, db=db)
#         logger.info(
#             f"Analysis complete — {request.freelancer_id}: "
#             f"score={result.portfolio_score}, skills={len(result.skills)}"
#         )
#         return result

#     except Exception as e:
#         logger.exception(f"Portfolio analysis failed for {request.freelancer_id}: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Analysis pipeline error: {str(e)}",
#         )

"""
app/api/endpoints/portfolio.py
────────────────────────────────
POST /analyze-portfolio

Accepts:
- PDF CV upload
- portfolio text
- GitHub URL
- portfolio URL

Extracts text from uploaded PDF and returns it.
"""

import fitz

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
)

from loguru import logger

from app.services.nlp.skill_extractor import extract_skills

from app.services.nlp.github_analyzer import (
    analyze_github_url,
    analyze_github_repo,
    _get_github_client,
    _parse_github_url,
)

from app.db.database import get_db

router = APIRouter()


@router.post(
    "/analyze-portfolio",
    status_code=status.HTTP_200_OK,
    summary="Analyze a freelancer portfolio",
    description="Upload a PDF CV or provide portfolio information for AI analysis.",
    tags=["Portfolio Analysis"],
)
async def analyze_portfolio_endpoint(
    freelancer_id: str = Form(None),
    file: UploadFile = File(None),
    portfolio_text: str = Form(None),
    github_url: str = Form(None),
    portfolio_url: str = Form(None),
    db=Depends(get_db),
):
    """
    Portfolio analysis endpoint.
    """

    # Validate at least one input exists
    if not any([file, portfolio_text, github_url, portfolio_url]):

        raise HTTPException(
            status_code=422,
            detail="Provide at least one input source",
        )

    # Validate uploaded file type
    if file and file.content_type != "application/pdf":

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed",
        )

    try:

        logger.info(
            f"Portfolio analysis request received "
            f"for freelancer_id={freelancer_id}"
        )

        extracted_text = ""

        # Read uploaded PDF
        if file:

            pdf_bytes = await file.read()

            logger.info(
                f"Uploaded file: {file.filename} "
                f"({len(pdf_bytes)} bytes)"
            )

            # Open PDF from memory
            pdf_document = fitz.open(
                stream=pdf_bytes,
                filetype="pdf"
            )

            # Extract text from all pages
            for page in pdf_document:
                extracted_text += page.get_text()

            pdf_document.close()

        # ── GitHub Analysis ────────────────────────────────────────────────
        github_analysis = None
        github_repo_analysis = []

        if github_url:

            github_analysis = analyze_github_url(github_url)

            username, repo_name = _parse_github_url(github_url)

            gh = _get_github_client()

            # If repo URL → analyze single repo
            if repo_name:

                repo_result = analyze_github_repo(
                    username=username,
                    repo_name=repo_name,
                    gh=gh,
                )

                github_repo_analysis.append(repo_result)

            # If profile URL → analyze all repos
            else:

                repo_names = github_analysis.get(
                    "repo_names",
                    []
                )

                for repo in repo_names:

                    repo_result = analyze_github_repo(
                        username=username,
                        repo_name=repo,
                        gh=gh,
                    )

                    github_repo_analysis.append(repo_result)

        return {

            "success": True,

            "freelancer_id": freelancer_id,

            "filename": (
                file.filename
                if file
                else None
            ),

            "portfolio_text": portfolio_text,

            "github_analysis": github_analysis,

            "github_repo_analysis": github_repo_analysis,

            "portfolio_url": portfolio_url,

            "skills": extract_skills(extracted_text),

            "extracted_text": extracted_text,
        }

    except Exception as e:

        logger.exception(f"Portfolio analysis failed: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline error: {str(e)}",
        )