"""Unit tests for ChatService orchestration (offline, mocked collaborators)."""

import pytest
from langchain_core.documents import Document

from app.ai.generation.citation_builder import CitationBuilder
from app.ai.memory.conversation_memory import ConversationMemory
from app.domain.chat.chat_service import ChatService


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    async def retrieve(self, query, k=None, **kwargs):
        return self._docs


class _FakeGenerator:
    def __init__(self):
        self.seen_history = "unset"

    async def generate(self, question, documents, history=None):
        self.seen_history = history
        return "answer citing [1]"


def _doc():
    return Document(
        page_content="context",
        metadata={"id": "d1", "title": "T", "path": "a.md", "score": 0.5},
    )


@pytest.mark.anyio
async def test_chat_returns_answer_and_citations():
    service = ChatService(
        retriever=_FakeRetriever([_doc()]),
        generator=_FakeGenerator(),
        citation_builder=CitationBuilder(),
        memory=ConversationMemory(),
    )

    result = await service.chat("q?")

    assert result.answer == "answer citing [1]"
    assert len(result.citations) == 1
    assert result.citations[0].index == 1
    assert result.citations[0].path == "a.md"


@pytest.mark.anyio
async def test_chat_persists_and_replays_history():
    generator = _FakeGenerator()
    memory = ConversationMemory()
    service = ChatService(
        retriever=_FakeRetriever([_doc()]),
        generator=generator,
        citation_builder=CitationBuilder(),
        memory=memory,
    )

    # First turn: no prior history yet (empty window).
    await service.chat("first?", conversation_id="c1")
    assert generator.seen_history == []

    # Second turn: prior turn is replayed into the generator.
    await service.chat("second?", conversation_id="c1")
    assert len(generator.seen_history) == 2  # one human + one ai message


@pytest.mark.anyio
async def test_stream_chat_yields_and_persists():
    class _StreamGen:
        async def stream(self, question, documents, history=None):
            for tok in ["a", "b", "c"]:
                yield tok

    memory = ConversationMemory()
    service = ChatService(
        retriever=_FakeRetriever([_doc()]),
        generator=_StreamGen(),
        citation_builder=CitationBuilder(),
        memory=memory,
    )

    tokens = [t async for t in service.stream_chat("q?", conversation_id="c2")]
    assert tokens == ["a", "b", "c"]
    assert len(memory.get_history("c2")) == 2
