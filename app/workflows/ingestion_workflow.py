"""
ingestion_workflow.py

LangGraph workflow for repository ingestion.

Graph:  ingest → index

``ingest`` clones the repo and parses files into KnowledgeDocuments;
``index`` chunks, embeds, and upserts them into the vector store.
"""

from langgraph.graph import END, START, StateGraph

from app.ai.pipelines.indexing_pipeline import IndexingPipeline
from app.core.config import settings
from app.domain.knowledge.services.ingestion_service import IngestionService
from app.workflows.state import IngestionState


class IngestionWorkflow:
    """Builds and runs the ingest → index LangGraph."""

    def __init__(
        self,
        ingestion_service: IngestionService,
        indexing_pipeline: IndexingPipeline,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._indexing_pipeline = indexing_pipeline
        self._graph = self._build()

    async def _ingest(self, state: IngestionState) -> IngestionState:
        documents = await self._ingestion_service.ingest_repository(
            state["repository_url"]
        )
        return {"documents": documents}

    async def _index(self, state: IngestionState) -> IngestionState:
        chunks = await self._indexing_pipeline.run(
            state.get("documents", []),
            collection_name=state.get("collection"),
        )
        return {"chunks_indexed": chunks}

    def _build(self):
        graph = StateGraph(IngestionState)
        graph.add_node("ingest", self._ingest)
        graph.add_node("index", self._index)
        graph.add_edge(START, "ingest")
        graph.add_edge("ingest", "index")
        graph.add_edge("index", END)
        return graph.compile()

    async def run(
        self,
        repository_url: str,
        collection: str | None = None,
    ) -> IngestionState:
        """Execute the graph and return the final IngestionState."""
        return await self._graph.ainvoke(
            {
                "repository_url": repository_url,
                "collection": collection or settings.DEFAULT_COLLECTION,
            }
        )
