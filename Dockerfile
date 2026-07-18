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

# Storage for cloned repos, uploads, and the local embedding model cache.
RUN mkdir -p storage/repositories storage/uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
