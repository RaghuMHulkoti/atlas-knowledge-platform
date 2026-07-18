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
    query: str = Field(min_length=1, description="Natural-language search query.")
    k: int | None = Field(
        default=None, ge=1, le=50, description="Maximum number of results."
    )
    repository: str | None = Field(
        default=None, description="Optional exact repository name to filter by."
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
    """Run a semantic search over the indexed knowledge base."""
    where = {"repository": request.repository} if request.repository else None

    results = await search_service.search(
        query=request.query,
        k=request.k,
        where=where,
    )

    return SearchResponse(
        query=request.query,
        count=len(results),
        results=results,
    )
