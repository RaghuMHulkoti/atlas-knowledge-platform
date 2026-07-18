"""
knowledge.py

Knowledge ingestion API endpoints.

The ingest endpoint runs the full synchronous pipeline: clone → parse →
chunk → embed → store. It returns both how many documents were parsed and how
many vector chunks were written to the store.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, HttpUrl

from app.ai.pipelines.indexing_pipeline import IndexingPipeline
from app.core.config import settings
from app.core.dependencies import (
    get_indexing_pipeline,
    get_ingestion_workflow,
    get_upload_service,
)
from app.core.exceptions import ConnectorException
from app.core.logging import get_logger
from app.domain.knowledge.services.upload_service import UploadService
from app.workflows.ingestion_workflow import IngestionWorkflow

logger = get_logger(__name__)

router = APIRouter()


class IngestRequest(BaseModel):
    repository_url: HttpUrl = Field(description="HTTPS Git URL to ingest.")
    collection: str | None = Field(
        default=None,
        description="Target collection name (defaults to the configured default).",
    )


class IngestResponse(BaseModel):
    repository_url: str
    collection: str
    documents_ingested: int
    chunks_indexed: int
    files: list[str]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_repository(
    request: IngestRequest,
    ingestion_workflow: Annotated[IngestionWorkflow, Depends(get_ingestion_workflow)],
) -> IngestResponse:
    """
    Clone a Git repository, convert every supported file into a
    KnowledgeDocument, then chunk, embed, and store those documents so they
    become searchable and chat-able.

    Orchestrated by the ingestion LangGraph workflow (ingest → index).
    """
    collection = request.collection or settings.DEFAULT_COLLECTION
    url = str(request.repository_url)

    state = await ingestion_workflow.run(repository_url=url, collection=collection)
    documents = state.get("documents", [])
    chunks_indexed = state.get("chunks_indexed", 0)

    logger.info(
        "Ingest complete: %s → %d document(s), %d chunk(s) in '%s'.",
        url,
        len(documents),
        chunks_indexed,
        collection,
    )

    return IngestResponse(
        repository_url=url,
        collection=collection,
        documents_ingested=len(documents),
        chunks_indexed=chunks_indexed,
        files=[doc.path for doc in documents],
    )


class UploadResponse(BaseModel):
    filename: str
    collection: str
    document_id: str
    chunks_indexed: int


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    upload_service: Annotated[UploadService, Depends(get_upload_service)],
    indexing_pipeline: Annotated[IndexingPipeline, Depends(get_indexing_pipeline)],
    file: Annotated[UploadFile, File(description="Document to index.")],
    collection: Annotated[str | None, Form()] = None,
) -> UploadResponse:
    """
    Upload a single document (PDF, DOCX, Markdown, or plain text), extract its
    text, then chunk, embed, and store it so it becomes searchable and
    chat-able.
    """
    target_collection = collection or settings.DEFAULT_COLLECTION
    filename = file.filename or "upload"

    data = await file.read()

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit.",
        )

    try:
        document = upload_service.build_document(
            data=data,
            filename=filename,
            collection=target_collection,
        )
    except ConnectorException as exc:
        # Unsupported type / empty text is a client error, not a 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks_indexed = await indexing_pipeline.run(
        [document], collection_name=target_collection
    )

    logger.info(
        "Upload complete: '%s' → %d chunk(s) in '%s'.",
        filename,
        chunks_indexed,
        target_collection,
    )

    return UploadResponse(
        filename=filename,
        collection=target_collection,
        document_id=document.id,
        chunks_indexed=chunks_indexed,
    )
