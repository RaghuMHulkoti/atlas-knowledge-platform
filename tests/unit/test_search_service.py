"""Unit tests for SearchService (offline, mocked retriever)."""

import pytest
from langchain_core.documents import Document

from app.domain.search.search_service import SearchService


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs
        self.last_kwargs = None

    async def retrieve(self, query, k=None, where=None, **kwargs):
        self.last_kwargs = {"query": query, "k": k, "where": where}
        return self._docs


@pytest.mark.anyio
async def test_search_maps_documents_to_results():
    docs = [
        Document(
            page_content="chunk text",
            metadata={
                "chunk_id": "c1",
                "id": "d1",
                "title": "Runbook",
                "source": "runbook.md",
                "repository": "atlas",
                "path": "runbook.md",
                "language": "markdown",
                "score": 0.9,
            },
        )
    ]
    retriever = _FakeRetriever(docs)
    service = SearchService(retriever=retriever)

    results = await service.search("restart worker", k=5)

    assert len(results) == 1
    r = results[0]
    assert r.document_id == "d1"
    assert r.title == "Runbook"
    assert r.repository == "atlas"
    assert r.score == 0.9
    assert r.snippet == "chunk text"
    assert retriever.last_kwargs["k"] == 5


@pytest.mark.anyio
async def test_search_empty_results():
    service = SearchService(retriever=_FakeRetriever([]))
    assert await service.search("nothing") == []
