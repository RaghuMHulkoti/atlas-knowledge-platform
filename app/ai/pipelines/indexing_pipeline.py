"""
indexing_pipeline.py

Orchestrates the full AI indexing pipeline for Atlas.

Pipeline:
    list[KnowledgeDocument]
        → DocumentConverter   (domain model → LangChain Document)
        → RecursiveSplitter   (full docs → fixed-size chunks)
        → EmbeddingService    (chunks → dense vectors)
        → BaseVectorStore     (upsert ids + text + embeddings + metadata)

This class contains zero business logic. Every step is delegated to a
dedicated collaborator. New steps can be inserted without modifying this class.
"""

import time

from langchain_core.documents import Document

from app.ai.converters.document_converter import DocumentConverter
from app.ai.embeddings.embedding_service import EmbeddingService
from app.ai.splitters.recursive_splitter import RecursiveSplitter
from app.core.config import settings
from app.core.exceptions import IndexingException
from app.core.logging import get_logger
from app.domain.knowledge.models import KnowledgeDocument
from app.infrastructure.vectorstore.base import BaseVectorStore

logger = get_logger(__name__)


class IndexingPipeline:
    """
    Orchestrates the conversion, chunking, embedding, and storage of
    KnowledgeDocuments into the vector store.

    Responsibilities:
    - Coordinate the four pipeline stages in sequence.
    - Log progress and elapsed time at each stage.
    - Wrap all failures in IndexingException.

    Non-responsibilities:
    - Any domain logic.
    - Git operations or file I/O (handled by IngestionService).
    - Chunking logic (RecursiveSplitter).
    - Embedding logic (EmbeddingService).
    - Vector store client details (BaseVectorStore).
    """

    def __init__(
        self,
        converter: DocumentConverter,
        splitter: RecursiveSplitter,
        embedding_service: EmbeddingService,
        vector_store: BaseVectorStore,
        collection_name: str | None = None,
    ) -> None:
        self._converter = converter
        self._splitter = splitter
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._collection_name = collection_name or settings.DEFAULT_COLLECTION

    async def run(
        self,
        documents: list[KnowledgeDocument],
        collection_name: str | None = None,
    ) -> int:
        """
        Run the full indexing pipeline for a batch of KnowledgeDocuments.

        Args:
            documents:       Raw domain documents from the ingestion pipeline.
            collection_name: Optional override for the target collection.
                             Defaults to the collection this pipeline was
                             constructed with.

        Returns:
            Total number of chunks stored in the vector store.

        Raises:
            IndexingException: If any pipeline stage fails.
        """
        collection = collection_name or self._collection_name

        if not documents:
            logger.info("IndexingPipeline: no documents to index.")
            return 0

        start = time.perf_counter()

        logger.info(
            "IndexingPipeline: starting — %d document(s) received.",
            len(documents),
        )

        try:
            # ── Stage 1: Convert KnowledgeDocuments → LangChain Documents ──
            lc_documents = self._converter.convert_many(documents)
            logger.info(
                "IndexingPipeline: stage 1 complete — %d document(s) converted.",
                len(lc_documents),
            )

            # ── Stage 2: Split into chunks ──
            chunks: list[Document] = self._splitter.split(lc_documents)
            logger.info(
                "IndexingPipeline: stage 2 complete — %d chunk(s) created.",
                len(chunks),
            )

            # ── Stage 3: Generate embeddings ──
            embeddings = self._embedding_service.embed(chunks)
            logger.info(
                "IndexingPipeline: stage 3 complete — %d embedding(s) generated.",
                len(embeddings),
            )

            # ── Stage 4: Upsert into vector store ──
            ids, texts, metas = self._prepare_upsert_payload(chunks)
            self._vector_store.upsert(
                collection_name=collection,
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metas,
            )

        except IndexingException:
            raise
        except Exception as exc:
            raise IndexingException(f"IndexingPipeline failed: {exc}") from exc

        elapsed = time.perf_counter() - start

        logger.info(
            "IndexingPipeline: complete — %d vector(s) stored in '%s' (%.2fs).",
            len(chunks),
            collection,
            elapsed,
        )

        return len(chunks)

    @staticmethod
    def _prepare_upsert_payload(
        chunks: list[Document],
    ) -> tuple[list[str], list[str], list[dict]]:
        """
        Build the parallel arrays required by BaseVectorStore.upsert().

        Each chunk ID is derived deterministically from its parent document ID
        plus a per-parent ordinal, so re-ingesting the same document overwrites
        its existing chunks instead of appending duplicates. Idempotent
        re-ingestion is a documented guarantee of the platform.

        Returns:
            (ids, texts, metadatas) — all lists of equal length.
        """
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []

        # Ordinal per parent document → stable chunk ids across re-ingestion.
        ordinal: dict[str, int] = {}

        for chunk in chunks:
            parent_id = chunk.metadata.get("id", "")
            index = ordinal.get(parent_id, 0)
            ordinal[parent_id] = index + 1
            chunk_id = f"{parent_id}_{index}"

            ids.append(chunk_id)
            texts.append(chunk.page_content)

            # ChromaDB only accepts str, int, float, bool metadata values
            clean_meta = {
                k: v
                for k, v in chunk.metadata.items()
                if isinstance(v, (str, int, float, bool))
            }
            metadatas.append(clean_meta)

        return ids, texts, metadatas
