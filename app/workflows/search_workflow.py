"""
search_workflow.py

LangGraph workflow for semantic search.

Graph:  retrieve → format

``retrieve`` fetches chunks via the injected retriever; ``format`` maps them
into transport-agnostic SearchResult value objects.
"""

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.ai.retrieval.base import BaseRetriever
from app.core.config import settings
from app.domain.search.search_service import SearchResult
from app.workflows.state import SearchState


class SearchWorkflow:
    """Builds and runs the semantic-search LangGraph."""

    def __init__(self, retriever: BaseRetriever) -> None:
        self._retriever = retriever
        self._graph = self._build()

    async def _retrieve(self, state: SearchState) -> SearchState:
        documents = await self._retriever.retrieve(
            query=state["query"],
            k=state.get("k") or settings.RETRIEVAL_TOP_K,
            where=state.get("where"),
        )
        return {"documents": documents}

    def _format(self, state: SearchState) -> SearchState:
        results = [_to_result(doc) for doc in state.get("documents", [])]
        return {"results": results}

    def _build(self):
        graph = StateGraph(SearchState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("format", self._format)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "format")
        graph.add_edge("format", END)
        return graph.compile()

    async def run(
        self,
        query: str,
        k: int | None = None,
        where: dict | None = None,
    ) -> SearchState:
        """Execute the graph and return the final SearchState."""
        return await self._graph.ainvoke(
            {
                "query": query,
                "k": k or settings.RETRIEVAL_TOP_K,
                "where": where,
            }
        )


def _to_result(doc: Document) -> SearchResult:
    meta = doc.metadata or {}
    return SearchResult(
        chunk_id=str(meta.get("chunk_id") or ""),
        document_id=str(meta.get("id") or ""),
        title=str(meta.get("title") or ""),
        source=str(meta.get("source") or ""),
        repository=str(meta.get("repository") or ""),
        path=str(meta.get("path") or ""),
        language=str(meta.get("language") or ""),
        score=meta.get("score"),
        snippet=doc.page_content,
    )
