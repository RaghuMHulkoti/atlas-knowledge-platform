"""
response_generator.py

Generates a grounded, citation-aware answer from a question and its retrieved
context. Owns prompt assembly and the LLM call; knows nothing about retrieval,
embeddings, or the vector store.
"""

from collections.abc import AsyncIterator

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.ai.prompts.templates import (
    LLM_UNAVAILABLE_MESSAGE,
    NO_CONTEXT_MESSAGE,
    RAG_SYSTEM_PROMPT,
    build_rag_user_prompt,
)
from app.core.logging import get_logger
from app.infrastructure.llm.base import BaseLLM

logger = get_logger(__name__)


class ResponseGenerator:
    """
    Turns (question, retrieved chunks, history) into an answer via the LLM.

    Responsibilities:
    - Assemble the system + history + grounded user prompt.
    - Invoke the LLM (single-shot or streaming).
    - Short-circuit to a safe message when there is no context.

    Non-responsibilities:
    - Retrieval or embeddings.
    - Building the citation payload (CitationBuilder).
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def _build_messages(
        self,
        question: str,
        documents: list[Document],
        history: list[BaseMessage] | None = None,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content=RAG_SYSTEM_PROMPT)]
        if history:
            messages.extend(history)
        messages.append(
            HumanMessage(content=build_rag_user_prompt(question, documents))
        )
        return messages

    async def generate(
        self,
        question: str,
        documents: list[Document],
        history: list[BaseMessage] | None = None,
    ) -> str:
        """
        Generate a complete answer grounded in *documents*.

        Returns a fixed no-context message when nothing was retrieved, so the
        model is never asked to answer without grounding.
        """
        if not documents:
            logger.info("ResponseGenerator: no context — returning safe message.")
            return NO_CONTEXT_MESSAGE

        messages = self._build_messages(question, documents, history)
        answer = await self._llm.generate_messages(messages)
        logger.info("ResponseGenerator: answer generated (%d chars).", len(answer))
        return answer

    async def stream(
        self,
        question: str,
        documents: list[Document],
        history: list[BaseMessage] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream the answer token-by-token.

        Falls back to yielding the whole no-context message when there is no
        grounding, and to a single-shot chunk if the LLM has no streaming API.
        """
        if not documents:
            yield NO_CONTEXT_MESSAGE
            return

        messages = self._build_messages(question, documents, history)

        stream_fn = getattr(self._llm, "stream_messages", None)
        if stream_fn is None:
            # Provider does not support streaming — emit the full answer once.
            yield await self._llm.generate_messages(messages)
            return

        # Stream resiliently: if generation fails before any token is emitted,
        # yield a friendly message instead of letting the exception break the
        # already-started HTTP stream (which surfaces as a raw traceback).
        emitted = False
        try:
            async for token in stream_fn(messages):
                emitted = True
                yield token
        except Exception:
            logger.exception("ResponseGenerator: streaming generation failed.")
            if not emitted:
                yield LLM_UNAVAILABLE_MESSAGE
