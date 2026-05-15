"""
app/services/nlp/pdf_extractor.py
───────────────────────────────────
PDF text extraction pipeline:
  1. Try PyMuPDF (fast, handles most PDFs)
  2. Fallback to pdfplumber (better table/layout handling)
  3. Final fallback: Tesseract OCR for scanned PDFs

Returns extracted text or raises an informative error.
"""

import base64
import io
import tempfile
import os
from pathlib import Path
from loguru import logger


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Extract text from raw PDF bytes.

    Returns:
        (text, method_used)  — method_used is one of: "pymupdf", "pdfplumber", "ocr"
    """
    # ── Attempt 1: PyMuPDF ────────────────────────────────────────────────────
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text("text"))
        doc.close()

        combined = "\n".join(text_parts).strip()
        if len(combined) > 100:      # Minimum viable text
            logger.debug(f"PyMuPDF extracted {len(combined)} chars")
            return combined, "pymupdf"
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e}")

    # ── Attempt 2: pdfplumber ─────────────────────────────────────────────────
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text_parts = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)

        combined = "\n".join(text_parts).strip()
        if len(combined) > 100:
            logger.debug(f"pdfplumber extracted {len(combined)} chars")
            return combined, "pdfplumber"
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")

    # ── Attempt 3: Tesseract OCR ──────────────────────────────────────────────
    logger.info("Falling back to Tesseract OCR for scanned PDF")
    return _ocr_extract(pdf_bytes)


def _ocr_extract(pdf_bytes: bytes) -> tuple[str, str]:
    """Convert PDF pages to images, then OCR each page."""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract

        images = convert_from_bytes(pdf_bytes, dpi=300)
        text_parts = []
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang="eng")
            text_parts.append(text)
            logger.debug(f"OCR page {i+1}: {len(text)} chars")

        combined = "\n".join(text_parts).strip()
        if combined:
            return combined, "ocr"
        raise ValueError("OCR produced no output")

    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise RuntimeError(f"All PDF extraction methods failed: {e}")


def extract_text_from_base64_pdf(b64_string: str) -> tuple[str, str]:
    """
    Decode a base64-encoded PDF and extract text.
    Handles both plain base64 and data-URI format.
    """
    # Strip data-URI prefix if present (data:application/pdf;base64,...)
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    try:
        pdf_bytes = base64.b64decode(b64_string)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {e}")

    return extract_text_from_pdf_bytes(pdf_bytes)