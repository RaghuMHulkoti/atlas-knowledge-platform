"""Unit tests for ResponseGenerator, CitationBuilder, and prompt templates."""

import pytest
from langchain_core.documents import Document

from app.ai.generation.citation_builder import CitationBuilder
from app.ai.generation.response_generator import ResponseGenerator
from app.ai.prompts.templates import (
    LLM_UNAVAILABLE_MESSAGE,
    NO_CONTEXT_MESSAGE,
    format_context,
)


class _CaptureLLM:
    def __init__(self):
        self.messages = None

    async def generate_messages(self, messages, **kwargs):
        self.messages = messages
        return "generated answer"


def _doc(text, **meta):
    return Document(page_content=text, metadata=meta)


@pytest.mark.anyio
async def test_generate_returns_no_context_message_when_empty():
    gen = ResponseGenerator(llm=_CaptureLLM())
    assert await gen.generate("q", documents=[]) == NO_CONTEXT_MESSAGE


@pytest.mark.anyio
async def test_stream_yields_message_when_llm_fails_before_output():
    """A streaming failure before any token yields a friendly message, not an error."""

    class _FailingLLM:
        async def stream_messages(self, messages, **kwargs):
            raise RuntimeError("all models down")
            yield  # pragma: no cover - makes this an async generator

    gen = ResponseGenerator(llm=_FailingLLM())
    tokens = [t async for t in gen.stream("q", [_doc("ctx", path="a.md")])]
    assert tokens == [LLM_UNAVAILABLE_MESSAGE]


@pytest.mark.anyio
async def test_generate_builds_grounded_prompt():
    llm = _CaptureLLM()
    gen = ResponseGenerator(llm=llm)

    answer = await gen.generate("How to restart?", [_doc("do X", path="run.md")])

    assert answer == "generated answer"
    # system + user
    assert len(llm.messages) == 2
    user_content = llm.messages[-1].content
    assert "How to restart?" in user_content
    assert "do X" in user_content
    assert "[1]" in user_content


def test_format_context_numbers_sources():
    ctx = format_context([_doc("alpha", path="a.md"), _doc("beta", path="b.md")])
    assert "[1] (a.md)" in ctx
    assert "[2] (b.md)" in ctx


def test_citation_builder_orders_and_truncates():
    long_text = "x" * 1000
    builder = CitationBuilder()
    citations = builder.build(
        [
            _doc("short", id="d1", title="T1", path="a.md", score=0.9),
            _doc(long_text, id="d2", title="T2", path="b.md"),
        ]
    )

    assert [c.index for c in citations] == [1, 2]
    assert citations[0].document_id == "d1"
    assert citations[1].snippet.endswith("…")
    assert len(citations[1].snippet) <= 402
