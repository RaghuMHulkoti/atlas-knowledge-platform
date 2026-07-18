"""
embedding_factory.py

Constructs the configured embedding provider.

Selects the concrete BaseEmbeddingProvider implementation based on the
EMBEDDING_PROVIDER application setting. This is the single place that decides
which embedding backend the platform uses — no other module names a concrete
provider.
"""

from app.ai.embeddings.base import BaseEmbeddingProvider
from app.core.config import settings
from app.core.exceptions import IndexingException
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingFactory:
    """
    Creates a BaseEmbeddingProvider from application settings.

    Supported providers:
    - "local"  -> LocalEmbeddingProvider  (on-device MiniLM; free, no API key)
    - "google" -> GoogleEmbeddingProvider (Gemini; requires GOOGLE_API_KEY)
    """

    @classmethod
    def create(cls, provider: str | None = None) -> BaseEmbeddingProvider:
        """
        Build the embedding provider named by *provider* (defaults to the
        EMBEDDING_PROVIDER setting).

        Raises:
            IndexingException: If the provider name is not recognised, or the
                "google" provider is selected without a GOOGLE_API_KEY.
        """
        name = (provider or settings.EMBEDDING_PROVIDER).strip().lower()

        if name == "local":
            # Imported lazily so the onnxruntime model is only loaded when the
            # local provider is actually selected.
            from app.ai.embeddings.local_provider import LocalEmbeddingProvider

            return LocalEmbeddingProvider()

        if name == "google":
            if settings.GOOGLE_API_KEY is None:
                raise IndexingException(
                    "EMBEDDING_PROVIDER='google' requires GOOGLE_API_KEY to be set."
                )
            from app.ai.embeddings.provider import GoogleEmbeddingProvider

            return GoogleEmbeddingProvider()

        raise IndexingException(
            f"Unknown EMBEDDING_PROVIDER '{name}'. Expected 'local' or 'google'."
        )
