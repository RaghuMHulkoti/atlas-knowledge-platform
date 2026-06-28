"""
knowledge.py

Knowledge ingestion API endpoints.
"""

from fastapi import APIRouter
from pydantic import BaseModel, HttpUrl

from app.domain.knowledge.services.ingestion_service import IngestionService
from app.infrastructure.connectors.git.connector import GitConnector
from app.infrastructure.connectors.git.loader import GitLoader
from app.infrastructure.connectors.git.parser import DocumentParser

router = APIRouter()


class IngestRequest(BaseModel):
    repository_url: HttpUrl


class IngestResponse(BaseModel):
    repository_url: str
    documents_ingested: int
    files: list[str]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_repository(request: IngestRequest):
    """
    Clone a Git repository and ingest all supported files
    into KnowledgeDocuments.
    """
    service = IngestionService(
        connector=GitConnector(),
        loader=GitLoader(),
        parser=DocumentParser(),
    )

    url = str(request.repository_url)
    documents = await service.ingest_repository(url)

    return IngestResponse(
        repository_url=url,
        documents_ingested=len(documents),
        files=[doc.path for doc in documents],
    )
