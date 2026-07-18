"""
dependencies.py

Central composition root for FastAPI dependency injection.

Every collaborator the API needs is constructed here and cached as a process
singleton via ``lru_cache``. Endpoints depend on these providers rather than on
concrete classes, so the wiring can change in one place.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.ai.converters.document_converter import DocumentConverter
from app.ai.embeddings.base import BaseEmbeddingProvider
from app.ai.embeddings.embedding_factory import EmbeddingFactory
from app.ai.embeddings.embedding_service import EmbeddingService
from app.ai.generation.citation_builder import CitationBuilder
from app.ai.generation.response_generator import ResponseGenerator
from app.ai.memory.conversation_memory import ConversationMemory
from app.ai.pipelines.indexing_pipeline import IndexingPipeline
from app.ai.retrieval.base import BaseRetriever
from app.ai.retrieval.chroma_retriever import ChromaRetriever
from app.ai.splitters.recursive_splitter import RecursiveSplitter
from app.core.config import Settings, get_settings
from app.domain.chat.chat_service import ChatService
from app.domain.knowledge.services.ingestion_service import IngestionService
from app.domain.knowledge.services.upload_service import UploadService
from app.domain.search.search_service import SearchService
from app.infrastructure.connectors.git.connector import GitConnector
from app.infrastructure.connectors.git.loader import GitLoader
from app.infrastructure.connectors.git.parser import DocumentParser
from app.infrastructure.llm.base import BaseLLM
from app.infrastructure.llm.openrouter import OpenRouterLLM
from app.infrastructure.vectorstore.base import BaseVectorStore
from app.infrastructure.vectorstore.chroma import ChromaVectorStore
from app.workflows.ingestion_workflow import IngestionWorkflow

SettingsDependency = Annotated[Settings, Depends(get_settings)]


# ----------------------------------------------------------------------
# Infrastructure singletons
# ----------------------------------------------------------------------


@lru_cache
def get_llm() -> BaseLLM:
    return OpenRouterLLM()


@lru_cache
def get_embedding_provider() -> BaseEmbeddingProvider:
    return EmbeddingFactory.create()


@lru_cache
def get_vector_store() -> BaseVectorStore:
    return ChromaVectorStore()


@lru_cache
def get_conversation_memory() -> ConversationMemory:
    return ConversationMemory()


# ----------------------------------------------------------------------
# AI components
# ----------------------------------------------------------------------


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(provider=get_embedding_provider())


@lru_cache
def get_retriever() -> BaseRetriever:
    return ChromaRetriever(
        vector_store=get_vector_store(),
        embedding_provider=get_embedding_provider(),
    )


@lru_cache
def get_indexing_pipeline() -> IndexingPipeline:
    return IndexingPipeline(
        converter=DocumentConverter(),
        splitter=RecursiveSplitter(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


# ----------------------------------------------------------------------
# Domain services
# ----------------------------------------------------------------------


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        connector=GitConnector(),
        loader=GitLoader(),
        parser=DocumentParser(),
    )


@lru_cache
def get_ingestion_workflow() -> IngestionWorkflow:
    return IngestionWorkflow(
        ingestion_service=get_ingestion_service(),
        indexing_pipeline=get_indexing_pipeline(),
    )


@lru_cache
def get_upload_service() -> UploadService:
    return UploadService()


@lru_cache
def get_search_service() -> SearchService:
    return SearchService(retriever=get_retriever())


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(
        retriever=get_retriever(),
        generator=ResponseGenerator(llm=get_llm()),
        citation_builder=CitationBuilder(),
        memory=get_conversation_memory(),
    )
