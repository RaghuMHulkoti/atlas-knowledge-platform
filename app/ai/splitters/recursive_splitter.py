"""
recursive_splitter.py

Splits LangChain Documents into smaller chunks using LangChain's
RecursiveCharacterTextSplitter.

Chunk size and overlap are read from application settings so they
can be tuned without code changes.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RecursiveSplitter:
    """
    Wraps LangChain's RecursiveCharacterTextSplitter.

    Responsibilities:
    - Split a list of LangChain Documents into fixed-size chunks.
    - Preserve all metadata from the parent document on every chunk.
    - Read chunk_size and chunk_overlap from application settings.

    Non-responsibilities:
    - Reading file content (DocumentParser).
    - Converting domain models (DocumentConverter).
    - Embedding or storing chunks.
    """

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            add_start_index=True,  # Adds 'start_index' to metadata for traceability
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split a batch of LangChain Documents into chunks.

        Metadata is automatically propagated to each chunk by LangChain's
        splitter — no manual copy is required.

        Args:
            documents: Full-length LangChain Documents from DocumentConverter.

        Returns:
            Flat list of chunk Documents, metadata-intact, ready for embedding.
        """
        if not documents:
            return []

        chunks = self._splitter.split_documents(documents)

        logger.info(
            "RecursiveSplitter: %d document(s) → %d chunk(s) "
            "(chunk_size=%d, overlap=%d)",
            len(documents),
            len(chunks),
            settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP,
        )

        return chunks
