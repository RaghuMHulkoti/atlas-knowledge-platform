from typing import Any

import chromadb

from app.core.config import settings


from app.infrastructure.vectorstore.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma Cloud implementation.
    """

    def __init__(self) -> None:
        self._client = chromadb.CloudClient(
            api_key=settings.CHROMA_API_KEY.get_secret_value(),
            tenant=settings.CHROMA_TENANT,
            database=settings.CHROMA_DATABASE,
        )

    @property
    def client(self):
        return self._client

    def get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(name=name)

    def get_collection(self, name: str):
        return self.client.get_collection(name=name)

    def delete_collection(self, name: str):
        self.client.delete_collection(name)

    def list_collections(self):
        return self.client.list_collections()

    def heartbeat(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """
        Upsert chunks with their embeddings into a ChromaDB collection.

        Uses collection.upsert() which creates or updates records by ID,
        making repeated ingestion of the same repository fully idempotent.
        """
        collection = self.get_or_create_collection(collection_name)
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        k: int,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the *k* nearest chunks to a pre-computed query embedding.

        The collection is created on demand so that querying a not-yet-indexed
        collection returns an empty list rather than raising.
        """
        collection = self.get_or_create_collection(collection_name)

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns each field as a list-of-lists (one inner list per
        # query embedding). We issue a single query, so index 0 throughout.
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        return [
            {
                "id": ids[i],
                "document": documents[i],
                "metadata": metadatas[i] or {},
                "distance": distances[i],
            }
            for i in range(len(ids))
        ]
