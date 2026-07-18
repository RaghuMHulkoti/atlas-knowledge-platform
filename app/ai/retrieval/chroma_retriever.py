"""
chroma_retriever.py

Semantic retriever backed by the Chroma vector store.

Embeds the incoming query with the same provider used at index time, then asks
the vector store for the nearest chunks. Query embeddings are computed
client-side so the query vector always matches the stored vectors' dimension.
"""

from langchain_core.documents import Document

from app.ai.embeddings.base import BaseEmbeddingProvider
from app.ai.retrieval.base import BaseRetriever
from app.core.config import settings
from app.core.exceptions import RetrievalException
from app.core.logging import get_logger
from app.infrastructure.vectorstore.base import BaseVectorStore

logger = get_logger(__name__)


class ChromaRetriever(BaseRetriever):
    """
    Retriever implementation utilizing the Chroma vector store.

    Responsibilities:
    - Embed the query via the injected embedding provider.
    - Delegate the nearest-neighbour lookup to the vector store.
    - Map raw store results into LangChain Documents, preserving provenance
      metadata and attaching the similarity distance/score.

    Non-responsibilities:
    - Embedding model details (BaseEmbeddingProvider).
    - Vector store client details (BaseVectorStore).
    - Prompt building or generation.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_provider: BaseEmbeddingProvider,
        collection_name: str | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._collection_name = collection_name or settings.DEFAULT_COLLECTION

    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        where: dict | None = None,
        **kwargs,
    ) -> list[Document]:
        """
        Retrieve the most relevant chunks for *query*.

        Args:
            query: Natural-language search string.
            k:     Maximum number of chunks to return (defaults to
                   settings.RETRIEVAL_TOP_K).
            where: Optional metadata filter passed through to the vector store.

        Returns:
            LangChain Documents, most relevant first. Each document's metadata
            carries the original chunk metadata plus ``distance`` and ``score``
            (``score = 1 - distance``, a convenience relevance value).

        Raises:
            RetrievalException: If embedding or the vector store query fails.
        """
        top_k = k or settings.RETRIEVAL_TOP_K

        if not query.strip():
            return []

        try:
            query_embedding = self._embedding_provider.embed_query(query)

            results = self._vector_store.query(
                collection_name=self._collection_name,
                query_embedding=query_embedding,
                k=top_k,
                where=where,
            )
        except Exception as exc:
            raise RetrievalException(
                f"Retrieval failed for query '{query[:60]}': {exc}"
            ) from exc

        documents: list[Document] = []
        for item in results:
            metadata = dict(item.get("metadata") or {})
            metadata["chunk_id"] = item.get("id")
            metadata["distance"] = item.get("distance")
            metadata["score"] = _to_score(item.get("distance"))

            documents.append(
                Document(
                    page_content=item.get("document") or "",
                    metadata=metadata,
                )
            )

        logger.info(
            "ChromaRetriever: %d chunk(s) retrieved for query (k=%d).",
            len(documents),
            top_k,
        )

        return documents


def _to_score(distance: float | None) -> float | None:
    """
    Convert a vector distance into a relevance score in (0, 1], higher = better.

    Uses ``1 / (1 + distance)`` so the score is always positive and monotonic in
    distance, regardless of the vector space's distance scale. (The previous
    ``1 - distance`` went negative for L2 distances > 1, which is every MiniLM
    result, and made score-thresholding clients drop all hits.)
    """
    if distance is None:
        return None
    return round(1.0 / (1.0 + float(distance)), 6)
