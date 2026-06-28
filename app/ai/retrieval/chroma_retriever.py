from typing import List

from langchain_core.documents import Document

from app.ai.retrieval.base import BaseRetriever
from app.infrastructure.vectorstore.base import BaseVectorStore


class ChromaRetriever(BaseRetriever):
    """
    Retriever implementation utilizing the Chroma Vector Store.
    """

    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store

    async def retrieve(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        return await self.vector_store.similarity_search(query=query, k=k, **kwargs)
