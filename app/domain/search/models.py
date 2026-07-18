"""
models.py

Value objects for the search domain.

Kept separate from the service and workflow so both can depend on it without a
circular import.
"""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single ranked search hit."""

    chunk_id: str = Field(default="", description="Vector store id of the chunk.")
    document_id: str = Field(default="", description="Parent KnowledgeDocument id.")
    title: str = Field(default="", description="Human-readable document title.")
    source: str = Field(default="", description="Origin of the document.")
    repository: str = Field(default="", description="Repository/collection name.")
    path: str = Field(default="", description="Relative path within the source.")
    language: str = Field(default="", description="Detected content language.")
    score: float | None = Field(
        default=None, description="Relevance score (higher is better)."
    )
    snippet: str = Field(default="", description="Matched chunk text.")
