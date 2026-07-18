"""
local_provider.py

On-device embedding provider — free, unlimited, no API key.

Uses ChromaDB's bundled ONNX model (all-MiniLM-L6-v2, 384-dim) via onnxruntime,
which ships with chromadb (no torch, no separate download service). The model
(~80 MB) is fetched once to a local cache on first use.

There is no quota and no rate limit, which makes this the right default for
ingesting large repositories.
"""

from chromadb.utils import embedding_functions

from app.ai.embeddings.base import BaseEmbeddingProvider
from app.core.logging import get_logger

logger = get_logger(__name__)


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider backed by an on-device ONNX MiniLM model.

    The same model embeds documents and queries, so write-time and query-time
    vectors are always dimensionally consistent (384-dim).
    """

    def __init__(self) -> None:
        self._embedding_function = embedding_functions.DefaultEmbeddingFunction()
        logger.info("LocalEmbeddingProvider initialised (all-MiniLM-L6-v2, 384-dim).")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document texts on-device.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of dense float vectors in the same order as input.
        """
        if not texts:
            return []

        # ChromaDB embedding functions return numpy arrays; coerce to plain
        # Python lists so the vectors are JSON-serialisable end to end.
        vectors = self._embedding_function(texts)
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string on-device.

        Args:
            text: Query string.

        Returns:
            A single dense float vector.
        """
        return self.embed_documents([text])[0]
