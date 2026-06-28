"""
provider.py

Google Generative AI embedding provider for the Atlas platform.

Implements BaseEmbeddingProvider using LangChain's GoogleGenerativeAIEmbeddings
integration. Configuration is read from application settings.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.ai.embeddings.base import BaseEmbeddingProvider
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider backed by Google Generative AI (text-embedding-004).

    Uses LangChain's integration — no manual HTTP calls.

    To swap providers (e.g. to OpenAI or Cohere), implement a new class
    that satisfies BaseEmbeddingProvider and inject it into EmbeddingService.
    """

    def __init__(self) -> None:
        self._model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
        )
        logger.info(
            "GoogleEmbeddingProvider initialised with model: %s",
            settings.EMBEDDING_MODEL,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document texts via Google Generative AI.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of dense float vectors in the same order as input.
        """
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string via Google Generative AI.

        Args:
            text: Query string.

        Returns:
            A single dense float vector.
        """
        return self._model.embed_query(text)
