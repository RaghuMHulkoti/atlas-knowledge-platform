"""
search.py

Semantic search API endpoints.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.dependencies import get_search_service
from app.domain.search.search_service import SearchResult, SearchService

router = APIRouter()


class SearchRequest(BaseModel):
    # Extra fields (e.g. a stray `repository`) are ignored rather than rejected,
    # so an over-eager client cannot accidentally 422 or over-filter the search.
    model_config = {"extra": "ignore"}

    query: str = Field(min_length=1, description="Natural-language search query.")
    k: int | None = Field(
        default=None, ge=1, le=50, description="Maximum number of results."
    )


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchResult]


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    search_service: Annotated[SearchService, Depends(get_search_service)],
) -> SearchResponse:
    """
    Run a semantic search over all indexed knowledge (the single default
    collection). No metadata filtering — every ingested repo/document is
    searched, so a query always sees everything that has been indexed.
    """
    results = await search_service.search(query=request.query, k=request.k)

    return SearchResponse(
        query=request.query,
        count=len(results),
        results=results,
    )
