"""Unit tests for the LangGraph workflows (offline, mocked collaborators)."""

import pytest
from langchain_core.documents import Document

from app.ai.generation.citation_builder import CitationBuilder
from app.ai.memory.conversation_memory import ConversationMemory
from app.workflows.chat_workflow import ChatWorkflow
from app.workflows.ingestion_workflow import IngestionWorkflow
from app.workflows.search_workflow import SearchWorkflow


class _Retriever:
    def __init__(self, docs):
        self._docs = docs

    async def retrieve(self, query, k=None, where=None, **kwargs):
        return self._docs


class _Generator:
    async def generate(self, question, documents, history=None):
        return f"answer[{len(documents)} ctx]"


def _doc():
    return Document(
        page_content="ctx",
        metadata={"id": "d1", "title": "T", "path": "a.md", "score": 0.5},
    )


@pytest.mark.anyio
async def test_chat_workflow_runs_all_nodes():
    memory = ConversationMemory()
    wf = ChatWorkflow(
        retriever=_Retriever([_doc()]),
        generator=_Generator(),
        citation_builder=CitationBuilder(),
        memory=memory,
    )

    state = await wf.run(question="q?", conversation_id="c1")

    assert state["answer"] == "answer[1 ctx]"
    assert len(state["citations"]) == 1
    assert state["citations"][0].path == "a.md"
    # cite node persisted the turn
    assert len(memory.get_history("c1")) == 2


@pytest.mark.anyio
async def test_search_workflow_formats_results():
    wf = SearchWorkflow(retriever=_Retriever([_doc()]))
    state = await wf.run(query="q", k=3)

    assert len(state["results"]) == 1
    assert state["results"][0].document_id == "d1"


@pytest.mark.anyio
async def test_ingestion_workflow_chains_ingest_and_index():
    class _Ingestion:
        async def ingest_repository(self, url):
            return ["docA", "docB"]

    class _Pipeline:
        def __init__(self):
            self.seen = None

        async def run(self, documents, collection_name=None):
            self.seen = (documents, collection_name)
            return len(documents)

    pipeline = _Pipeline()
    wf = IngestionWorkflow(ingestion_service=_Ingestion(), indexing_pipeline=pipeline)

    state = await wf.run(repository_url="http://x", collection="atlas")

    assert state["chunks_indexed"] == 2
    assert pipeline.seen == (["docA", "docB"], "atlas")
