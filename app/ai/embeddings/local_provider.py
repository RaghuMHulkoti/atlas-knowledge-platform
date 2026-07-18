"""
local_provider.py

On-device embedding provider backed by sentence-transformers.

Loads a HuggingFace model (default: BAAI/bge-large-en-v1.5, 1024-dim) and runs
it locally — no API key required. The model is downloaded once to the local
HuggingFace cache on first use.

BGE retrieval models expect an instruction prefix on the *query* (not on the
indexed passages); this is applied in ``embed_query`` and configured via
``EMBEDDING_QUERY_INSTRUCTION``. Vectors are L2-normalized so cosine and
Euclidean rankings agree with Chroma's default space.
"""

from sentence_transformers import SentenceTransformer

from app.ai.embeddings.base import BaseEmbeddingProvider
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider backed by an on-device sentence-transformers model.

    The same model embeds both documents and queries, so write-time and
    query-time vectors are always the same dimensionality.
    """

    def __init__(self) -> None:
        self._model_name = settings.EMBEDDING_MODEL
        self._query_instruction = settings.EMBEDDING_QUERY_INSTRUCTION or ""
        self._model = SentenceTransformer(self._model_name)
        dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "LocalEmbeddingProvider initialised (model=%s, dim=%s).",
            self._model_name,
            dimension,
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [list(map(float, vector)) for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document/passage texts on-device.

        No instruction prefix is applied — BGE expects passages to be embedded
        as-is.
        """
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query on-device, applying the retrieval instruction
        prefix so the query vector lands near relevant passages.
        """
        return self._encode([f"{self._query_instruction}{text}"])[0]
