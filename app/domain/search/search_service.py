"""
search_service.py

Semantic search over indexed knowledge.

Thin domain facade over the search LangGraph workflow. Retrieval and result
shaping happen inside the workflow's nodes; this service exposes a simple async
``search`` method and returns the ranked SearchResult objects.
"""

from app.ai.retrieval.base import BaseRetriever
from app.core.config import settings
from app.core.logging import get_logger
from app.domain.search.models import SearchResult
from app.workflows.search_workflow import SearchWorkflow

logger = get_logger(__name__)

# Re-exported for backwards compatibility with existing imports.
__all__ = ["SearchResult", "SearchService"]


class SearchService:
    """
    Executes semantic search via the SearchWorkflow.

    Responsibilities:
    - Run the search workflow and return its ranked SearchResult objects.

    Non-responsibilities:
    - Embedding / vector store details (retriever, inside the workflow).
    - Answer generation (ChatService).
    """

    def __init__(self, retriever: BaseRetriever) -> None:
        self._workflow = SearchWorkflow(retriever=retriever)

    async def search(
        self,
        query: str,
        k: int | None = None,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """
        Run a semantic search.

        Args:
            query: Natural-language query.
            k:     Maximum number of results (defaults to RETRIEVAL_TOP_K).
            where: Optional metadata filter.

        Returns:
            Ranked list of SearchResult, most relevant first.
        """
        top_k = k or settings.RETRIEVAL_TOP_K
        logger.info("SearchService: query=%r k=%d", query[:80], top_k)

        state = await self._workflow.run(query=query, k=top_k, where=where)
        results = state.get("results", [])

        logger.info("SearchService: returning %d result(s).", len(results))
        return results
