"""Tests for the background ingestion job store and the async /ingest flow."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.core.dependencies import get_ingestion_workflow, get_job_store
from app.domain.knowledge.jobs import JobStatus, JobStore
from app.main import app


def test_job_store_lifecycle():
    store = JobStore()
    job = store.create("http://x", "atlas")
    assert job.status == JobStatus.PENDING
    assert store.get(job.id) is not None

    store.update(job.id, status=JobStatus.COMPLETED, chunks_indexed=5)
    updated = store.get(job.id)
    assert updated.status == JobStatus.COMPLETED
    assert updated.chunks_indexed == 5

    assert store.get("missing") is None


class _FakeWorkflow:
    async def run(self, repository_url, collection):
        return {
            "documents": [
                SimpleNamespace(path="src/a.py"),
                SimpleNamespace(path="b.md"),
            ],
            "chunks_indexed": 7,
        }


class _FailingWorkflow:
    async def run(self, repository_url, collection):
        raise RuntimeError("clone exploded")


def test_ingest_returns_202_and_job_completes():
    store = JobStore()
    app.dependency_overrides[get_ingestion_workflow] = _FakeWorkflow
    app.dependency_overrides[get_job_store] = lambda: store
    try:
        client = TestClient(app)
        resp = client.post(
            "/api/v1/knowledge/ingest",
            json={"repository_url": "https://github.com/octocat/Spoon-Knife.git"},
        )
        assert resp.status_code == 202
        body = resp.json()
        job_id = body["job_id"]
        assert body["status"] == "pending"
        assert body["status_url"].endswith(job_id)

        # TestClient runs the background task before returning, so by now the
        # job has finished.
        status = client.get(f"/api/v1/knowledge/jobs/{job_id}")
        assert status.status_code == 200
        job = status.json()
        assert job["status"] == "completed"
        assert job["documents_ingested"] == 2
        assert job["chunks_indexed"] == 7
        assert job["files"] == ["src/a.py", "b.md"]
    finally:
        app.dependency_overrides.clear()


def test_ingest_job_records_failure():
    store = JobStore()
    app.dependency_overrides[get_ingestion_workflow] = _FailingWorkflow
    app.dependency_overrides[get_job_store] = lambda: store
    try:
        client = TestClient(app)
        resp = client.post(
            "/api/v1/knowledge/ingest",
            json={"repository_url": "https://github.com/octocat/Spoon-Knife.git"},
        )
        job_id = resp.json()["job_id"]
        job = client.get(f"/api/v1/knowledge/jobs/{job_id}").json()
        assert job["status"] == "failed"
        assert "clone exploded" in job["error"]
    finally:
        app.dependency_overrides.clear()


def test_unknown_job_returns_404():
    client = TestClient(app)
    assert client.get("/api/v1/knowledge/jobs/nope").status_code == 404
