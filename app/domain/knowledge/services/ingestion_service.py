"""
ingestion_service.py

Orchestrates the Git repository ingestion pipeline.

This service coordinates three collaborators:
  - BaseConnector  — acquires the repository from a remote source
  - GitLoader      — discovers ingestible files within the local clone
  - DocumentParser — converts each file into a KnowledgeDocument

The service contains no Git internals, no file I/O, and no parsing logic.
It only composes its collaborators and reports progress via structured logs.
"""

from app.core.logging import get_logger
from app.domain.knowledge.models import KnowledgeDocument
from app.infrastructure.connectors.base import BaseConnector
from app.infrastructure.connectors.git.loader import GitLoader
from app.infrastructure.connectors.git.parser import DocumentParser

logger = get_logger(__name__)


class IngestionService:
    """
    Orchestrates the ingestion pipeline for a Git repository.

    Responsibilities:
    - Coordinate clone → discover → parse.
    - Log pipeline progress and totals.
    - Return the complete list of KnowledgeDocuments to the caller.

    Non-responsibilities:
    - Git operations (BaseConnector).
    - File system traversal (GitLoader).
    - File I/O or language detection (DocumentParser).
    - Chunking, embedding, or vector store writes (future sprints).
    """

    def __init__(
        self,
        connector: BaseConnector,
        loader: GitLoader,
        parser: DocumentParser,
    ) -> None:
        self._connector = connector
        self._loader = loader
        self._parser = parser

    async def ingest_repository(
        self,
        repository_url: str,
    ) -> list[KnowledgeDocument]:
        """
        Run the full ingestion pipeline for a single Git repository.

        Steps:
          1. Clone (or pull) the repository via the connector.
          2. Discover all ingestible files via the loader.
          3. Parse each file into a KnowledgeDocument via the parser.
          4. Return all successfully parsed documents.

        Args:
            repository_url: The remote Git URL to ingest.

        Returns:
            List of KnowledgeDocument instances, one per successfully parsed file.
            Files that cannot be read are logged and silently skipped.
        """
        logger.info("Starting ingestion for repository: %s", repository_url)

        # 1. Acquire the repository locally (clone or pull).
        repo_path = self._connector.clone(repository_url)

        # 2. Discover all ingestible file paths.
        file_paths = self._loader.load(repo_path)

        logger.info(
            "Discovered %d file(s) to process in '%s'.",
            len(file_paths),
            repo_path.name,
        )

        # 3. Parse each file into a KnowledgeDocument.
        documents: list[KnowledgeDocument] = []
        skipped = 0

        for file_path in file_paths:
            document = self._parser.parse(
                file_path=file_path,
                repo_path=repo_path,
                source_url=repository_url,
            )

            if document is None:
                skipped += 1
                continue

            documents.append(document)

        logger.info(
            "Ingestion complete for '%s': %d document(s) created, %d file(s) skipped.",
            repo_path.name,
            len(documents),
            skipped,
        )

        return documents
