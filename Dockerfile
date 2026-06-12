# ─────────────────────────────────────────────────────────────────────────────
# Masar AI Microservice — Dockerfile
# Multi-stage build for lean production image
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies needed for build (not kept in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into /build/deps
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/build/deps --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Masar Team"
LABEL description="Masar AI Microservice"

WORKDIR /app

# System runtime deps: Tesseract OCR + Poppler (pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /build/deps /usr/local

# Download spaCy model (baked into image for faster cold starts)
RUN python -m spacy download en_core_web_sm

# Copy application source
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/faiss_index /app/logs

# Non-root user for security
RUN adduser --disabled-password --gecos "" masaruser \
    && chown -R masaruser:masaruser /app
USER masaruser

# Expose FastAPI port
# EXPOSE 8000

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#     CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" \
#     || exit 1

# # Start server
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()" \
    || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]