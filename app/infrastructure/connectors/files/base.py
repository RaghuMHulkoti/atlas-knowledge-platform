"""
base.py

Contract for file parsers used by the document-upload pipeline.

A file parser extracts plain text from the raw bytes of a single uploaded file
of one specific type (PDF, DOCX, Markdown, plain text). It knows nothing about
chunking, embedding, or the KnowledgeDocument model.
"""

from abc import ABC, abstractmethod


class BaseFileParser(ABC):
    """
    Extracts plain text from the raw bytes of one file type.

    Implementations declare which file extensions they handle and how to turn
    bytes into text. Everything else (ids, metadata, chunking) is handled by the
    upload service downstream.
    """

    #: File extensions this parser handles, lower-case, dot-prefixed (".pdf").
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def extract_text(self, data: bytes, filename: str) -> str:
        """
        Extract plain text from *data*.

        Args:
            data:     Raw file bytes.
            filename: Original filename (used only for diagnostics).

        Returns:
            The extracted text. May be empty if the file has no text content.
        """
        raise NotImplementedError
