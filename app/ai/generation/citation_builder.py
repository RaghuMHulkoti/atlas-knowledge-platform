"""
citation_builder.py

Builds the citation list that accompanies a generated answer.

Citations are numbered to match the numbered context block produced by
``prompts.templates.format_context`` — source [1] in the prompt corresponds to
Citation index 1 here — so the numbers the model emits line up with the
metadata returned to the caller.
"""

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single cited source backing a generated answer."""

    index: int = Field(description="1-based citation number used in the answer.")
    document_id: str = Field(default="", description="Parent KnowledgeDocument id.")
    title: str = Field(default="", description="Human-readable document title.")
    source: str = Field(default="", description="Origin of the document.")
    repository: str = Field(default="", description="Repository/collection name.")
    path: str = Field(default="", description="Relative path within the source.")
    score: float | None = Field(default=None, description="Relevance score.")
    snippet: str = Field(default="", description="Chunk text that was cited.")


class CitationBuilder:
    """
    Converts retrieved chunks into an ordered list of Citations.

    The ordering must match the order the chunks were rendered into the prompt,
    so pass the exact same list used to build the context block.
    """

    #: Snippets longer than this are truncated in the citation payload.
    _MAX_SNIPPET_CHARS = 400

    def build(self, documents: list[Document]) -> list[Citation]:
        """Build 1-based citations from retrieved documents, in order."""
        citations: list[Citation] = []
        for index, doc in enumerate(documents, start=1):
            meta = doc.metadata or {}
            citations.append(
                Citation(
                    index=index,
                    document_id=str(meta.get("id") or ""),
                    title=str(meta.get("title") or ""),
                    source=str(meta.get("source") or ""),
                    repository=str(meta.get("repository") or ""),
                    path=str(meta.get("path") or ""),
                    score=meta.get("score"),
                    snippet=self._truncate(doc.page_content),
                )
            )
        return citations

    def _truncate(self, text: str) -> str:
        text = text.strip()
        if len(text) <= self._MAX_SNIPPET_CHARS:
            return text
        return text[: self._MAX_SNIPPET_CHARS].rstrip() + "…"
