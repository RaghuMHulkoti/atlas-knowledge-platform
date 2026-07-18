# syntax=docker/dockerfile:1

# ==============================================================================
# Atlas - Enterprise Engineering Knowledge Platform
# Production image built with uv on a slim Python base.
# ==============================================================================

FROM python:3.11-slim AS base

# - git:  required by the Git connector to clone repositories at runtime.
# - curl: used by the container HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv binary from the official distroless image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# ------------------------------------------------------------------------------
# Dependency layer — cached unless the lockfile changes.
# ------------------------------------------------------------------------------
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ------------------------------------------------------------------------------
# Application layer.
# ------------------------------------------------------------------------------
COPY app ./app
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Storage for cloned repos and uploads.
RUN mkdir -p storage/repositories storage/uploads

# Pre-download the on-device embedding model (all-MiniLM-L6-v2, ~80 MB) so it is
# baked into the image. Without this the model downloads on the first ingest
# request, making that request slow enough to hit a gateway 502.
RUN python -c "from chromadb.utils import embedding_functions; embedding_functions.DefaultEmbeddingFunction()(['warmup'])"

EXPOSE 8000

# start-period is generous so the one-time warm-up (model load + Chroma client)
# finishes before health checks start counting failures.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
