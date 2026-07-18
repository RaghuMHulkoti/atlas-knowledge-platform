"""
embedding_service.py

Generates embeddings for a batch of LangChain Documents.

Accepts any BaseEmbeddingProvider so the underlying model can be swapped
without changing this service or the IndexingPipeline.
"""

from langchain_core.documents import Document

from app.ai.embeddings.base import BaseEmbeddingProvider
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Generates dense vector embeddings for a list of LangChain Documents.

    Responsibilities:
    - Extract page_content from each document.
    - Delegate embedding generation to BaseEmbeddingProvider.
    - Return embeddings in the same order as the input documents.

    Non-responsibilities:
    - Writing to any vector store.
    - Chunking or splitting documents.
    - Any business logic.
    """

    def __init__(self, provider: BaseEmbeddingProvider) -> None:
        self._provider = provider

    def embed(self, documents: list[Document]) -> list[list[float]]:
        """
        Generate embeddings for a batch of LangChain Documents.

        Args:
            documents: Chunks produced by RecursiveSplitter.

        Returns:
            List of dense float vectors, one per document, in the same order.
        """
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)

        logger.info(
            "EmbeddingService: generating embeddings for %d chunk(s) "
            "(batch_size=%d).",
            len(texts),
            batch_size,
        )

        # Embed in bounded batches. Passing every chunk to the model at once can
        # spike memory far above the model's baseline (enough to be OOM-killed
        # in a memory-limited container); batching keeps the peak flat.
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings.extend(self._provider.embed_documents(batch))
            logger.debug(
                "EmbeddingService: embedded %d/%d chunk(s).",
                len(embeddings),
                len(texts),
            )

        logger.info(
            "EmbeddingService: %d embedding(s) generated.",
            len(embeddings),
        )

        return embeddings
