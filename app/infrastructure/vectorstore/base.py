from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """
    Contract for every vector store implementation.
    """

    @abstractmethod
    def get_or_create_collection(self, name: str) -> Any:
        pass

    @abstractmethod
    def get_collection(self, name: str) -> Any:
        pass

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        pass

    @abstractmethod
    def list_collections(self) -> list[Any]:
        pass

    @abstractmethod
    def heartbeat(self) -> bool:
        pass

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Upsert documents with their embeddings into a collection.

        Idempotent: if an id already exists, its record is updated.

        Args:
            collection_name: Target collection.
            ids:             Unique identifier per document/chunk.
            documents:       Raw text content per document/chunk.
            embeddings:      Dense vector per document/chunk.
            metadatas:       Metadata dict per document/chunk.
        """
        pass

    @abstractmethod
    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the *k* nearest chunks to a pre-computed query embedding.

        Embeddings are computed client-side (same as ``upsert``); the query
        vector must have the same dimensionality as the stored vectors.

        Args:
            collection_name: Collection to search.
            query_embedding: Dense query vector.
            k:               Maximum number of results to return.
            where:           Optional ChromaDB metadata filter.

        Returns:
            A list of result dicts, nearest first, each shaped as::

                {"id": str, "document": str, "metadata": dict, "distance": float}
        """
        pass
