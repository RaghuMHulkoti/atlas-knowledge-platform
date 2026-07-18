"""
knowledge.py

Knowledge ingestion API endpoints.

Repository ingestion (clone -> parse -> chunk -> embed -> store) can take longer
than an HTTP gateway will wait, so it runs as a BACKGROUND JOB: POST /ingest
returns 202 with a job id immediately, and the client polls GET /jobs/{id} for
progress. Single-file uploads are small and stay synchronous.
"""

import asyncio
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field, HttpUrl

from app.ai.pipelines.indexing_pipeline import IndexingPipeline
from app.core.config import settings
from app.core.dependencies import (
    get_indexing_pipeline,
    get_ingestion_workflow,
    get_job_store,
    get_upload_service,
)
from app.core.exceptions import ConnectorException
from app.core.logging import get_logger
from app.domain.knowledge.jobs import IngestionJob, JobStatus, JobStore
from app.domain.knowledge.services.upload_service import UploadService
from app.workflows.ingestion_workflow import IngestionWorkflow

logger = get_logger(__name__)

router = APIRouter()


class IngestRequest(BaseModel):
    repository_url: HttpUrl = Field(description="HTTPS Git URL to ingest.")


class IngestJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    collection: str
    status_url: str


def _run_ingestion_job(
    workflow: IngestionWorkflow,
    job_store: JobStore,
    job_id: str,
    url: str,
    collection: str,
) -> None:
    """
    Execute an ingestion in the background and record its outcome.

    Runs as a Starlette background task (sync -> threadpool), so the blocking
    embed/upsert work never holds the HTTP request open. ``asyncio.run`` drives
    the async ingestion workflow on this worker thread.
    """
    job_store.update(job_id, status=JobStatus.RUNNING)
    logger.info("Ingest job %s started: %s -> '%s'", job_id, url, collection)

    try:
        state = asyncio.run(workflow.run(repository_url=url, collection=collection))
        documents = state.get("documents", [])
        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            documents_ingested=len(documents),
            chunks_indexed=state.get("chunks_indexed", 0),
            files=[doc.path for doc in documents],
        )
        logger.info(
            "Ingest job %s complete: %d document(s), %d chunk(s).",
            job_id,
            len(documents),
            state.get("chunks_indexed", 0),
        )
    except Exception as exc:
        logger.exception("Ingest job %s failed.", job_id)
        job_store.update(job_id, status=JobStatus.FAILED, error=str(exc))


@router.post("/ingest", status_code=202, response_model=IngestJobResponse)
async def ingest_repository(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    ingestion_workflow: Annotated[IngestionWorkflow, Depends(get_ingestion_workflow)],
    job_store: Annotated[JobStore, Depends(get_job_store)],
) -> IngestJobResponse:
    """
    Start ingesting a Git repository in the background.

    Returns immediately with a job id (HTTP 202). The repository is cloned,
    parsed, chunked, embedded, and stored asynchronously; poll the returned
    ``status_url`` (GET /knowledge/jobs/{id}) for progress and final counts.

    Everything is indexed into the single default collection
    (``settings.DEFAULT_COLLECTION``) so all knowledge lives in one place.
    """
    collection = settings.DEFAULT_COLLECTION
    url = str(request.repository_url)

    job = job_store.create(repository_url=url, collection=collection)
    background_tasks.add_task(
        _run_ingestion_job,
        ingestion_workflow,
        job_store,
        job.id,
        url,
        collection,
    )

    logger.info("Ingest job %s queued for %s.", job.id, url)

    return IngestJobResponse(
        job_id=job.id,
        status=job.status,
        collection=collection,
        status_url=f"{settings.API_V1_PREFIX}/knowledge/jobs/{job.id}",
    )


@router.get("/jobs/{job_id}", response_model=IngestionJob)
async def get_ingestion_job(
    job_id: str,
    job_store: Annotated[JobStore, Depends(get_job_store)],
) -> IngestionJob:
    """Return the status and results of a background ingestion job."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job id '{job_id}'.")
    return job


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
) -> UploadResponse:
    """
    Upload a single document (PDF, DOCX, Markdown, or plain text), extract its
    text, then chunk, embed, and store it so it becomes searchable and
    chat-able.

    Indexed into the single default collection (``settings.DEFAULT_COLLECTION``).
    """
    target_collection = settings.DEFAULT_COLLECTION
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
