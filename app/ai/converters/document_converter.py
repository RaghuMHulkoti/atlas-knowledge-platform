"""
document_converter.py

Converts the Atlas canonical KnowledgeDocument into a LangChain Document.

This is the boundary layer between the domain model (KnowledgeDocument) and
the AI pipeline (LangChain). Everything downstream of this converter speaks
LangChain — nothing upstream needs to know about LangChain internals.
"""

from langchain_core.documents import Document

from app.domain.knowledge.models import KnowledgeDocument


class DocumentConverter:
    """
    Converts KnowledgeDocument → langchain_core.documents.Document.

    Responsibilities:
    - Translate the canonical domain model into the LangChain wire format.
    - Preserve all provenance fields in the LangChain metadata dict.

    Non-responsibilities:
    - Chunking, splitting, or truncating content.
    - Embedding or vector store writes.
    - Any I/O.
    """

    def convert(self, document: KnowledgeDocument) -> Document:
        """
        Convert a single KnowledgeDocument into a LangChain Document.

        The ``page_content`` field carries the full raw text. All provenance
        and domain fields are flattened into ``metadata`` so they survive
        through the splitter and are stored alongside each chunk in ChromaDB.

        Args:
            document: The canonical domain model to convert.

        Returns:
            A LangChain Document ready for splitting and embedding.
        """
        metadata = {
            # Provenance
            "id": document.id,
            "source": document.source,
            "repository": document.repository,
            "path": document.path,
            "title": document.title,
            "language": document.language,
            # Pass-through domain metadata (file_name, size_bytes, etc.)
            **document.metadata,
        }

        # Exclude None values — ChromaDB rejects None metadata values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return Document(
            page_content=document.content,
            metadata=metadata,
        )

    def convert_many(self, documents: list[KnowledgeDocument]) -> list[Document]:
        """
        Convert a batch of KnowledgeDocuments.

        Args:
            documents: List of canonical domain models.

        Returns:
            List of LangChain Documents in the same order.
        """
        return [self.convert(doc) for doc in documents]
