"""
jobs.py

In-process job tracking for asynchronous ingestion.

Ingesting a large repository (clone -> parse -> embed -> upsert) can take longer
than an HTTP gateway will wait, causing 502s. To avoid that, ingestion runs in
the background and its progress is tracked here so the client can poll for it.

This store is in-memory and per-process: jobs do not survive a restart and are
not shared across replicas. That is sufficient for a single-instance deployment;
back it with Redis/DB when scaling horizontally.
"""

import uuid
from enum import Enum
from threading import Lock

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionJob(BaseModel):
    """State of a single background ingestion."""

    id: str
    status: JobStatus = JobStatus.PENDING
    repository_url: str
    collection: str
    documents_ingested: int = 0
    chunks_indexed: int = 0
    files: list[str] = Field(default_factory=list)
    error: str | None = None


class JobStore:
    """Thread-safe registry of ingestion jobs keyed by id."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: dict[str, IngestionJob] = {}

    def create(self, repository_url: str, collection: str) -> IngestionJob:
        """Register a new PENDING job and return it."""
        job = IngestionJob(
            id=uuid.uuid4().hex,
            repository_url=repository_url,
            collection=collection,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> IngestionJob | None:
        """Return the job with *job_id*, or None if unknown."""
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        """Apply *fields* to the stored job (no-op if the job is gone)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            self._jobs[job_id] = job.model_copy(update=fields)
