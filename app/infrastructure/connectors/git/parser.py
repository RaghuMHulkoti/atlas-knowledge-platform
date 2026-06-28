"""
parser.py

Converts a raw file Path into a canonical KnowledgeDocument.

This is the only place in the ingestion pipeline that reads file content.
All other stages (connector, loader, service) remain decoupled from file I/O.
"""

import hashlib
from pathlib import Path

from app.core.logging import get_logger
from app.domain.knowledge.models import KnowledgeDocument

logger = get_logger(__name__)

# Maps file extensions to human-readable language identifiers used downstream
# by the chunker to select the appropriate splitting strategy.
_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".kt": "kotlin",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".sql": "sql",
    ".properties": "properties",
    ".gradle": "gradle",
}


def _make_document_id(source_url: str, relative_path: str) -> str:
    """
    Generate a deterministic, collision-resistant document ID.

    The ID is a SHA-256 hex digest of the concatenation of the source URL
    and the file's relative path within the repository. This ensures that
    re-ingesting the same file from the same source always produces the
    same ID, enabling idempotent upserts in ChromaDB.
    """
    raw = f"{source_url}::{relative_path}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _derive_title(file_path: Path) -> str:
    """
    Derive a human-readable title from the file path.

    Uses the file stem (filename without extension) as the title.
    Example: 'docs/architecture.md' → 'architecture'
    """
    return file_path.stem


def _derive_repository_name(source_url: str) -> str:
    """
    Extract a repository name from a URL.

    Example:
        https://github.com/org/atlas.git  →  atlas
    """
    name = source_url.rstrip("/").split("/")[-1]
    return name.removesuffix(".git") or source_url


class DocumentParser:
    """
    Converts a file Path into a KnowledgeDocument.

    Responsibilities:
    - Read file content safely (UTF-8 with replacement for bad bytes).
    - Detect programming/natural language from file extension.
    - Generate a deterministic document ID.
    - Populate all KnowledgeDocument fields derivable from the filesystem.

    Non-responsibilities:
    - Discovering which files to parse (GitLoader).
    - Chunking or embedding content.
    - Any network I/O.
    """

    def parse(
        self,
        file_path: Path,
        repo_path: Path,
        source_url: str,
    ) -> KnowledgeDocument | None:
        """
        Parse a single file into a KnowledgeDocument.

        Args:
            file_path:  Absolute path to the file.
            repo_path:  Absolute path to the repository root.
                        Used to compute the relative path stored in the document.
            source_url: Remote URL of the repository (e.g. GitHub HTTPS URL).
                        Used for provenance and ID generation.

        Returns:
            A populated KnowledgeDocument, or None if the file cannot be read.
        """
        try:
            relative_path = str(file_path.relative_to(repo_path))
        except ValueError:
            logger.warning(
                "File '%s' is not inside repo '%s', skipping.",
                file_path,
                repo_path,
            )
            return None

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning(
                "Could not read file '%s': %s — skipping.",
                file_path,
                exc,
            )
            return None

        language = _EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower(), "")
        document_id = _make_document_id(source_url, relative_path)
        repository = _derive_repository_name(source_url)
        title = _derive_title(file_path)

        document = KnowledgeDocument(
            id=document_id,
            source=source_url,
            repository=repository,
            path=relative_path,
            title=title,
            content=content,
            language=language,
            metadata={
                "file_name": file_path.name,
                "file_extension": file_path.suffix.lower(),
                "size_bytes": file_path.stat().st_size,
            },
        )

        logger.debug(
            "Parsed document: id=%s path=%s language=%s",
            document_id[:12],
            relative_path,
            language,
        )

        return document
