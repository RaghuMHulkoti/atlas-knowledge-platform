"""
knowledge_document.py

Canonical domain model for a document ingested into the Atlas Knowledge Platform.

This is the single source of truth for a knowledge artifact from the moment it is
loaded from a source (Git, PDF, DOCX, Markdown, plain text) until it is embedded
and stored in ChromaDB. Every stage of the AI pipeline — Chunking, Embedding,
Indexing, Retrieval, and Generation — traces its provenance back to this model.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeDocument(BaseModel):
    """
    Canonical representation of a knowledge artifact in the Atlas platform.

    Immutable by design: once a document is loaded from a source it must not be
    mutated in-place. Downstream pipeline stages (chunker, embedder, indexer)
    produce their own output types rather than modifying this object.

    Fields are intentionally broad (plain Python types) so that the model can
    be serialised to JSON, stored as ChromaDB metadata, or passed through
    LangGraph state without any conversion layer.
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable value object — no in-place mutation
        populate_by_name=True,  # Accept both alias and field name
        str_strip_whitespace=True,  # Trim accidental leading/trailing whitespace
        extra="forbid",  # Reject unknown fields at construction time
    )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    id: str = Field(
        description=(
            "Globally unique identifier for this document. "
            "Typically a deterministic hash of (source, path, commit_hash) "
            "so that re-ingestion of the same content is idempotent."
        ),
    )

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    source: str = Field(
        description=(
            "The origin of this document. "
            "For Git sources this is the remote URL; "
            "for file uploads it is the original filename or storage path."
        ),
    )

    repository: str = Field(
        default="",
        description=(
            "Human-readable name of the repository or collection this document "
            "belongs to (e.g. 'atlas-knowledge-platform'). "
            "Empty string when the document does not originate from a VCS."
        ),
    )

    path: str = Field(
        default="",
        description=(
            "Relative file path within the repository or upload directory "
            "(e.g. 'docs/architecture.md'). "
            "Empty string for documents that have no meaningful path."
        ),
    )

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    title: str = Field(
        default="",
        description=(
            "Human-readable title extracted or inferred from the document. "
            "Used as a display label in search results and the UI."
        ),
    )

    content: str = Field(
        description="Full raw text content of the document before chunking.",
    )

    language: str = Field(
        default="",
        description=(
            "Detected or declared programming/natural language of the content "
            "(e.g. 'python', 'en', 'markdown'). "
            "Used by the chunker to apply language-aware splitting strategies."
        ),
    )

    # ------------------------------------------------------------------
    # Metadata  (pass-through dict for ChromaDB filters & LLM context)
    # ------------------------------------------------------------------

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary key-value pairs that are passed through the entire pipeline "
            "and stored alongside each chunk in ChromaDB. "
            "Typical keys: 'author', 'file_type', 'size_bytes', 'mime_type'. "
            "Values must be JSON-serialisable (str, int, float, bool, or None)."
        ),
    )

    # ------------------------------------------------------------------
    # Version control  (optional — populated for Git-sourced documents)
    # ------------------------------------------------------------------

    branch: str | None = Field(
        default=None,
        description="Git branch from which this document was loaded.",
    )

    commit_hash: str | None = Field(
        default=None,
        description="Full SHA-1 / SHA-256 Git commit hash at time of ingestion.",
    )

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    checksum: str | None = Field(
        default=None,
        description=(
            "SHA-256 hex digest of the raw file bytes. "
            "Used for deduplication and change detection during re-ingestion."
        ),
    )

    # ------------------------------------------------------------------
    # Temporal
    # ------------------------------------------------------------------

    created_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when this document was first ingested.",
    )

    updated_at: datetime | None = Field(
        default=None,
        description="UTC timestamp of the most recent re-ingestion of this document.",
    )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Free-form labels for filtering and grouping documents "
            "(e.g. ['architecture', 'api', 'onboarding'])."
        ),
    )
