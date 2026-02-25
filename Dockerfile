# ---------------------------------------------------------------------------
# Premier League Predictor - Production Dockerfile
# ---------------------------------------------------------------------------
# Two-stage approach:
#   1) "builder" installs Python deps (keeps final image lean)
#   2) "runtime" copies only what's needed to run the API
# ---------------------------------------------------------------------------

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create data directories (match your repo structure)
RUN mkdir -p data/raw data/processed data/historical data/predictions models

# Non-root user for security
RUN adduser --disabled-password --no-create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Uvicorn with sensible production defaults
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
