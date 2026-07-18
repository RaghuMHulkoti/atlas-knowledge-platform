"""HTTP-path tests for /chat and /chat/stream using a stub-LLM-backed service."""

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.ai.generation.citation_builder import CitationBuilder
from app.ai.generation.response_generator import ResponseGenerator
from app.ai.memory.conversation_memory import ConversationMemory
from app.core.dependencies import get_chat_service
from app.domain.chat.chat_service import ChatService
from app.infrastructure.llm.base import BaseLLM
from app.main import app


class _StubLLM(BaseLLM):
    async def generate(self, prompt, **kwargs):
        return "ok"

    async def generate_messages(self, messages, **kwargs):
        return "Run make deploy [1]."

    async def stream_messages(self, messages, **kwargs):
        for token in ["Run ", "make ", "deploy ", "[1]."]:
            yield token

    def count_tokens(self, text):
        return 0

    def provider_name(self):
        return "stub"

    def model_name(self):
        return "stub"

    async def health_check(self):
        return True


class _StubRetriever:
    async def retrieve(self, query, k=None, **kwargs):
        return [
            Document(
                page_content="deploy with make deploy",
                metadata={
                    "id": "d1",
                    "title": "Runbook",
                    "path": "run.md",
                    "score": 0.9,
                },
            )
        ]


def _stub_chat_service():
    return ChatService(
        retriever=_StubRetriever(),
        generator=ResponseGenerator(llm=_StubLLM()),
        citation_builder=CitationBuilder(),
        memory=ConversationMemory(),
    )


def setup_module(module):
    app.dependency_overrides[get_chat_service] = _stub_chat_service


def teardown_module(module):
    app.dependency_overrides.clear()


def test_chat_endpoint_returns_answer_and_citations():
    client = TestClient(app)
    resp = client.post("/api/v1/chat", json={"question": "how do I deploy?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Run make deploy [1]."
    assert len(body["citations"]) == 1
    assert body["citations"][0]["index"] == 1
    assert body["citations"][0]["path"] == "run.md"


def test_chat_stream_endpoint_streams_tokens():
    client = TestClient(app)
    resp = client.post("/api/v1/chat/stream", json={"question": "how do I deploy?"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "Run make deploy [1]."
