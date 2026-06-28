from app.infrastructure.vectorstore.chroma import ChromaVectorStore
from typing import Annotated
from functools import lru_cache

from fastapi import Depends
from langchain_core.embeddings import Embeddings

from app.core.config import Settings, get_settings
from app.infrastructure.llm.base import BaseLLM
from app.infrastructure.llm.openrouter import OpenRouterLLM
from app.ai.embeddings.embedding_service import GoogleEmbeddingsService
from app.infrastructure.vectorstore.base import BaseVectorStore

# from app.infrastructure.vectorstore.chroma import ChromaStore
from app.ai.retrieval.base import BaseRetriever
from app.ai.retrieval.chroma_retriever import ChromaRetriever

SettingsDependency = Annotated[
    Settings,
    Depends(get_settings),
]


@lru_cache
def get_llm() -> BaseLLM:
    return OpenRouterLLM()


@lru_cache
def get_embeddings() -> Embeddings:
    return GoogleEmbeddingsService().get_embeddings()


# @lru_cache
# def get_vector_store() -> BaseVectorStore:
#     # We use a default collection name 'atlas' as a placeholder
#     return ChromaStore(
#         embeddings=get_embeddings(),
#         collection_name="atlas"
#     )


@lru_cache
def get_retriever() -> BaseRetriever:
    return ChromaRetriever(vector_store=get_vector_store())


@lru_cache
def get_vector_store() -> BaseVectorStore:
    return ChromaVectorStore()
