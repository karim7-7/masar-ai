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

import io
import pytesseract
from PIL import Image

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
from app.services.nlp.experience_detector import analyze_experience_and_complexity

from app.services.nlp.preprocessor import preprocess_portfolio_text
from app.services.nlp.portfolio_scorer import analyze_portfolio_quality


from app.models.mongo_models import freelancer_profile_doc, portfolio_analysis_doc


from app.services.embedding.embedding_service import (
    store_embedding,
    build_freelancer_text,
)
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
    allowed_types = [
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
]

    if file and file.content_type not in allowed_types:

        raise HTTPException(
        status_code=400,
        detail="Only PDF or image files are allowed",
    )

    try:

        logger.info(
            f"Portfolio analysis request received "
            f"for freelancer_id={freelancer_id}"
        )

        extracted_text = ""

        # Read uploaded PDF
        if file:

            file_bytes = await file.read()

            # PDF
            if file.content_type == "application/pdf":
                pdf_document = fitz.open(
                stream=file_bytes,
                filetype="pdf"
            )

                for page in pdf_document:
                    extracted_text += page.get_text()

            pdf_document.close()

        # Images OCR
        else:
            image = Image.open(io.BytesIO(file_bytes))

            extracted_text = pytesseract.image_to_string(image)

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


        # ── Combine all available text ─────────────────────────────
        # combined_text = extracted_text or ""

        # if portfolio_text:
        #     combined_text += "\n" + portfolio_text

                # ── Combine all available text ─────────────────────────────
        combined_text = extracted_text or ""

        if portfolio_text:
            combined_text += "\n" + portfolio_text

        if github_analysis:
            if github_analysis.get("description"):
                combined_text += "\n" + github_analysis.get("description", "")

            if github_analysis.get("readme_text"):
                combined_text += "\n" + github_analysis.get("readme_text", "")

        for repo_result in github_repo_analysis:
            if repo_result.get("description"):
                combined_text += "\n" + repo_result.get("description", "")

            if repo_result.get("readme_text"):
                combined_text += "\n" + repo_result.get("readme_text", "")

        if not combined_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No readable text found for analysis",
            )

        if not combined_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No readable text found for analysis",
            )

        # # ── Skill Extraction ───────────────────────────────────────
        # skills = extract_skills(combined_text)

        # # ── Experience Analysis ────────────────────────────────────
        # preprocessed = {
        #     "cleaned_text": combined_text,
        #     "years_of_experience": 0,
        # }

        # experience_analysis = analyze_experience_and_complexity(
        #     preprocessed,
        #     skills,
        # )

                # ── Preprocess Text ────────────────────────────────────────
        preprocessed = preprocess_portfolio_text(combined_text)

        # ── Skill Extraction ───────────────────────────────────────
        skills = extract_skills(preprocessed["cleaned_text"])

        # ── GitHub verified skills boost ───────────────────────────
        github_verified_skills = []

        if github_analysis:
            github_verified_skills.extend(
                github_analysis.get("verified_skills", [])
            )

        for repo_result in github_repo_analysis:
            github_verified_skills.extend(
                repo_result.get("verified_skills", [])
            )

        github_verified_lower = {
            skill.lower() for skill in github_verified_skills
        }

        for skill in skills:
            if skill["name"].lower() in github_verified_lower:
                skill["confidence"] = min(skill["confidence"] + 0.07, 0.99)
                skill["verified"] = True
                skill["source"] = "github+nlp"

        existing_skill_names = {skill["name"].lower() for skill in skills}

        for github_skill in github_verified_skills:
            if github_skill.lower() not in existing_skill_names:
                skills.append(
                    {
                        "name": github_skill,
                        "confidence": 0.88,
                        "category": "programming_languages",
                        "verified": True,
                        "source": "github",
                    }
                )

        # ── Experience Analysis ────────────────────────────────────
        experience_analysis = analyze_experience_and_complexity(
            preprocessed,
            skills,
        )



                # ── Portfolio Quality Scoring ──────────────────────────────
        portfolio_quality = analyze_portfolio_quality(
            preprocessed,
            skills,
            experience_analysis,
        )

        verified_skills = list(
            set(
                portfolio_quality.get("verified_skills", [])
                + github_verified_skills
            )
        )
                # ── Save Freelancer Profile for Matching ───────────────────
        if freelancer_id:
            verified_skills = []

            if github_analysis:
                verified_skills = github_analysis.get("verified_skills", [])

            profile_doc = freelancer_profile_doc(
                freelancer_id=str(freelancer_id),
                name="",
                email="",
                skills=skills,
                experience_level=experience_analysis.get(
                    "experience_level",
                    "Beginner",
                ),
                years_of_experience=experience_analysis.get(
                    "years_of_experience",
                    0,
                ),
                portfolio_url=portfolio_url or "",
                github_url=github_url or "",
            )

            await db["freelancer_profiles"].replace_one(
                {"freelancer_id": str(freelancer_id)},
                profile_doc,
                upsert=True,
            )

            # analysis_doc = portfolio_analysis_doc(
            #     freelancer_id=str(freelancer_id),
            #     skills=skills,
            #     verified_skills=verified_skills,
            #     experience_level=experience_analysis.get(
            #         "experience_level",
            #         "Beginner",
            #     ),
            #     years_of_experience=experience_analysis.get(
            #         "years_of_experience",
            #         0,
            #     ),
            #     project_complexity=experience_analysis.get(
            #         "project_complexity",
            #         "Simple",
            #     ),
            #     portfolio_score=0.0,
            #     technical_depth_score=0.0,
            #     project_realism_score=0.0,
            #     raw_text_length=len(combined_text),
            #     source="pdf" if file else "manual",
            # )

            # await db["portfolio_analyses"].replace_one(
            #     {"freelancer_id": str(freelancer_id)},
            #     analysis_doc,
            #     upsert=True,
            # )

            analysis_doc = portfolio_analysis_doc(
            freelancer_id=str(freelancer_id),
            skills=skills,
            verified_skills=verified_skills,
            experience_level=experience_analysis.get(
                    "experience_level",
                    "Beginner",
                ),
            years_of_experience=experience_analysis.get(
                    "years_of_experience",
                    0,
                ),
            project_complexity=experience_analysis.get(
                    "project_complexity",
                    "Simple",
                ),
            portfolio_score=portfolio_quality.get("portfolio_score", 0.0),
            technical_depth_score=portfolio_quality.get(
                    "technical_depth_score",
                    0.0,
                ),
            project_realism_score=portfolio_quality.get(
                    "project_realism_score",
                    0.0,
                ),
            raw_text_length=preprocessed.get("char_count", len(combined_text)),
            source="pdf" if file else "manual",
            )
        # ── Build Freelancer Text for Embedding ────────────────────
        verified_skills = []

        if github_analysis:
            verified_skills = github_analysis.get("verified_skills", [])

        profile_for_embedding = {
            "bio": portfolio_text or "",
            "skills": skills,
            "experience_level": experience_analysis.get("experience_level"),
            "years_of_experience": experience_analysis.get("years_of_experience"),
            "portfolio_description": combined_text[:3000],
            "verified_skills": verified_skills,
            "portfolio_score": portfolio_quality.get("portfolio_score", 0.0),
            "technical_depth_score": portfolio_quality.get(
                "technical_depth_score",
                0.0,
            ),
            "project_realism_score": portfolio_quality.get(
                "project_realism_score",
                0.0,
            ),
        }
        

        embedding_text = build_freelancer_text(profile_for_embedding)

        # ── Generate and Store Embedding ───────────────────────────
        embedding_info = None

        if freelancer_id and embedding_text:
            embedding, faiss_id = await store_embedding(
                entity_id=str(freelancer_id),
                entity_type="freelancer",
                text=embedding_text,
                db=db,
            )

            embedding_info = {
                "embedding_dimension": len(embedding),
                "faiss_index_id": faiss_id,
                "message": "Embedding stored successfully",
            }

        # ── Final Response ─────────────────────────────────────────
            return {
            "success": True,
            "freelancer_id": freelancer_id,
            "filename": file.filename if file else None,
            "portfolio_text": portfolio_text,
            "portfolio_url": portfolio_url,
            "github_analysis": github_analysis,
            "github_repo_analysis": github_repo_analysis,
            "skills": skills,
            "verified_skills": verified_skills,
            "experience_analysis": experience_analysis,
            "portfolio_quality": portfolio_quality,
            "embedding": embedding_info,
            "saved_to_mongodb": bool(freelancer_id),
        }

        # return {

        #     "success": True,

        #     "freelancer_id": freelancer_id,

        #     "filename": (
        #         file.filename
        #         if file
        #         else None
        #     ),

        #     "portfolio_text": portfolio_text,

        #     "github_analysis": github_analysis,

        #     "github_repo_analysis": github_repo_analysis,

        #     "portfolio_url": portfolio_url,

        #     "skills": extract_skills(extracted_text),

        #     "experience_analysis": analyze_experience_and_complexity({"cleaned_text": extracted_text}, extract_skills(extracted_text)),
        # }

    except Exception as e:

        logger.exception(f"Portfolio analysis failed: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline error: {str(e)}",
        )