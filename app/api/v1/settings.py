"""
settings.py

Read-only introspection of the running configuration.

Exposes only non-sensitive settings — API keys and other secrets are never
returned.
"""

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("")
async def get_runtime_settings() -> dict:
    """Return the non-sensitive runtime configuration."""
    return {
        "application": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
        },
        "llm": {
            "provider": settings.LLM_PROVIDER,
            "primary_model": settings.LLM_PRIMARY_MODEL,
            "fallback_models": settings.fallback_models,
        },
        "embeddings": {
            "provider": settings.EMBEDDING_PROVIDER,
            "model": (
                settings.GOOGLE_EMBEDDING_MODEL
                if settings.EMBEDDING_PROVIDER == "google"
                else "all-MiniLM-L6-v2"
            ),
        },
        "vector_store": {
            "type": "chroma-cloud",
            "default_collection": settings.DEFAULT_COLLECTION,
        },
        "retrieval": {
            "top_k": settings.RETRIEVAL_TOP_K,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
        },
    }
