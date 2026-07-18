"""
upload_service.py

Converts an uploaded file into a KnowledgeDocument.

Selects the right parser for the file type, extracts text, and builds a
canonical KnowledgeDocument with the same id/provenance conventions used by the
Git ingestion path — so uploaded and repository-sourced documents flow through
the identical indexing pipeline.
"""

import hashlib
from pathlib import Path

from app.core.exceptions import ConnectorException
from app.core.logging import get_logger
from app.domain.knowledge.models import KnowledgeDocument
from app.infrastructure.connectors.files.factory import FileParserFactory

logger = get_logger(__name__)


class UploadService:
    """
    Builds a KnowledgeDocument from an uploaded file's bytes.

    Responsibilities:
    - Dispatch to the correct file parser by extension.
    - Extract text and compute a deterministic id + checksum.
    - Produce a KnowledgeDocument ready for the indexing pipeline.

    Non-responsibilities:
    - Chunking, embedding, or storage (IndexingPipeline).
    - HTTP concerns (the API layer).
    """

    def build_document(
        self,
        data: bytes,
        filename: str,
        collection: str,
    ) -> KnowledgeDocument:
        """
        Parse *data* and construct a KnowledgeDocument.

        Args:
            data:       Raw uploaded file bytes.
            filename:   Original filename (drives type detection & metadata).
            collection: Target collection, recorded as the document repository.

        Returns:
            A populated KnowledgeDocument.

        Raises:
            ConnectorException: If the file type is unsupported or has no text.
        """
        parser = FileParserFactory.for_filename(filename)
        content = parser.extract_text(data, filename)

        if not content.strip():
            raise ConnectorException(f"No extractable text found in '{filename}'.")

        checksum = hashlib.sha256(data).hexdigest()
        # id is content-addressed by (filename, checksum) so re-uploading the
        # same bytes is idempotent, matching the Git path's deterministic ids.
        document_id = hashlib.sha256(f"{filename}::{checksum}".encode()).hexdigest()

        extension = Path(filename).suffix.lower()

        document = KnowledgeDocument(
            id=document_id,
            source=filename,
            repository=collection,
            path=filename,
            title=Path(filename).stem,
            content=content,
            language=_extension_to_language(extension),
            checksum=checksum,
            metadata={
                "file_name": filename,
                "file_extension": extension,
                "size_bytes": len(data),
                "ingestion_source": "upload",
            },
        )

        logger.info(
            "UploadService: built document id=%s from '%s' (%d chars).",
            document_id[:12],
            filename,
            len(content),
        )
        return document


def _extension_to_language(extension: str) -> str:
    return {
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".pdf": "pdf",
        ".docx": "docx",
    }.get(extension, "")
