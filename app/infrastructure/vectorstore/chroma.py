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
