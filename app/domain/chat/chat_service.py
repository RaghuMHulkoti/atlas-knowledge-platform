"""
chat_service.py

Conversational retrieval-augmented generation.

Facade over the chat LangGraph workflow (load_history → retrieve → generate →
cite). Non-streaming answers run through the graph; streaming answers use the
generator directly, since token streaming does not fit a node-at-a-time graph.
"""

from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

from app.ai.generation.citation_builder import Citation, CitationBuilder
from app.ai.generation.response_generator import ResponseGenerator
from app.ai.memory.conversation_memory import ConversationMemory
from app.ai.retrieval.base import BaseRetriever
from app.core.config import settings
from app.core.logging import get_logger
from app.workflows.chat_workflow import ChatWorkflow

logger = get_logger(__name__)


class ChatResult(BaseModel):
    """The result of a single chat turn."""

    answer: str = Field(description="Grounded, citation-annotated answer.")
    citations: list[Citation] = Field(
        default_factory=list, description="Sources backing the answer, in order."
    )
    conversation_id: str | None = Field(
        default=None, description="Conversation id for multi-turn continuity."
    )


class ChatService:
    """
    Runs conversational RAG over the indexed knowledge base.

    The non-streaming path is orchestrated by ChatWorkflow; the collaborators
    below are also held directly to serve the streaming path.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: ResponseGenerator,
        citation_builder: CitationBuilder,
        memory: ConversationMemory,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._memory = memory
        self._workflow = ChatWorkflow(
            retriever=retriever,
            generator=generator,
            citation_builder=citation_builder,
            memory=memory,
        )

    async def chat(
        self,
        question: str,
        conversation_id: str | None = None,
        k: int | None = None,
    ) -> ChatResult:
        """
        Answer *question*, grounded in retrieved context and prior turns.

        Runs the chat workflow and maps its final state to a ChatResult.
        """
        state = await self._workflow.run(
            question=question,
            conversation_id=conversation_id,
            k=k or settings.RETRIEVAL_TOP_K,
        )

        citations = state.get("citations", [])
        logger.info(
            "ChatService: answered (conversation_id=%s, %d citation(s)).",
            conversation_id,
            len(citations),
        )

        return ChatResult(
            answer=state.get("answer", ""),
            citations=citations,
            conversation_id=conversation_id,
        )

    async def stream_chat(
        self,
        question: str,
        conversation_id: str | None = None,
        k: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream the answer token-by-token, then persist the full turn.

        Yields answer text chunks as they are produced. Citations are not part
        of the token stream; callers that need them should use ``chat`` or query
        ``/search`` alongside.
        """
        top_k = k or settings.RETRIEVAL_TOP_K
        history = self._memory.get_history(conversation_id) if conversation_id else None

        documents = await self._retriever.retrieve(query=question, k=top_k)

        collected: list[str] = []
        async for token in self._generator.stream(
            question=question,
            documents=documents,
            history=history,
        ):
            collected.append(token)
            yield token

        if conversation_id:
            self._memory.add_turn(conversation_id, question, "".join(collected))
