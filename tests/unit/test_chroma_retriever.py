"""Unit tests for ChromaRetriever (offline, fully mocked)."""

import pytest

from app.ai.retrieval.chroma_retriever import ChromaRetriever
from app.core.exceptions import RetrievalException


class _FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeStore:
    def __init__(self, results=None, raises=False):
        self._results = results or []
        self._raises = raises
        self.last_call = None

    def query(self, collection_name, query_embedding, k, where=None):
        if self._raises:
            raise RuntimeError("boom")
        self.last_call = {
            "collection_name": collection_name,
            "k": k,
            "where": where,
        }
        return self._results


@pytest.mark.anyio
async def test_retrieve_maps_results_to_documents():
    store = _FakeStore(
        results=[
            {
                "id": "chunk-1",
                "document": "hello world",
                "metadata": {"id": "doc-1", "title": "T", "path": "a.md"},
                "distance": 0.25,
            }
        ]
    )
    retriever = ChromaRetriever(
        vector_store=store,
        embedding_provider=_FakeEmbeddings(),
        collection_name="atlas",
    )

    docs = await retriever.retrieve("what is x?", k=3)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.page_content == "hello world"
    assert doc.metadata["chunk_id"] == "chunk-1"
    assert doc.metadata["distance"] == 0.25
    # score = 1 / (1 + distance) = 1 / 1.25 = 0.8
    assert doc.metadata["score"] == pytest.approx(0.8)
    assert store.last_call["collection_name"] == "atlas"
    assert store.last_call["k"] == 3


@pytest.mark.anyio
async def test_retrieve_empty_query_returns_empty():
    store = _FakeStore(results=[{"id": "x"}])
    retriever = ChromaRetriever(store, _FakeEmbeddings())
    assert await retriever.retrieve("   ") == []


@pytest.mark.anyio
async def test_retrieve_wraps_store_errors():
    retriever = ChromaRetriever(_FakeStore(raises=True), _FakeEmbeddings())
    with pytest.raises(RetrievalException):
        await retriever.retrieve("query")
