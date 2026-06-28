"""
base.py

Abstract contract for embedding providers.

Decouples the embedding service from any specific provider (Google, OpenAI,
Cohere, local HuggingFace, etc.). Swapping providers requires only a new
implementation of this interface — the EmbeddingService and IndexingPipeline
are never touched.
"""

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """
    Contract for all embedding providers.

    Both methods must return vectors in the same order as the input texts.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of document texts.

        Args:
            texts: List of raw text strings to embed.

        Returns:
            List of dense float vectors, one per input text, in the same order.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single query string.

        Separated from embed_documents because some providers use different
        models or prefixes for asymmetric retrieval (query vs. passage).

        Args:
            text: The query string to embed.

        Returns:
            A single dense float vector.
        """
        raise NotImplementedError
