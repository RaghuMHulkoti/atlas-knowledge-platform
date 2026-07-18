"""
state.py

Typed state objects passed between LangGraph workflow nodes.

Each workflow declares the exact keys its nodes read and write. Using TypedDict
(rather than a Pydantic model) keeps state cheap to copy and merge as LangGraph
threads it through the graph.
"""

from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class ChatState(TypedDict, total=False):
    """State for the conversational RAG workflow."""

    question: str
    conversation_id: str | None
    k: int
    history: list[BaseMessage] | None
    documents: list[Document]
    answer: str
    citations: list[Any]


class SearchState(TypedDict, total=False):
    """State for the semantic search workflow."""

    query: str
    k: int
    where: dict[str, Any] | None
    documents: list[Document]
    results: list[Any]


class IngestionState(TypedDict, total=False):
    """State for the repository ingestion + indexing workflow."""

    repository_url: str
    collection: str
    documents: list[Any]
    chunks_indexed: int
