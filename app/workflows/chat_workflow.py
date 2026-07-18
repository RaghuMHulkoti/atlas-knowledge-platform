"""
chat_workflow.py

LangGraph workflow for conversational RAG.

Graph:  load_history → retrieve → generate → cite

Each node is a small, single-responsibility step that reads and writes the
shared ChatState. The concrete collaborators (retriever, generator, citation
builder, memory) are injected, so the graph owns orchestration only.
"""

from langgraph.graph import END, START, StateGraph

from app.ai.generation.citation_builder import CitationBuilder
from app.ai.generation.response_generator import ResponseGenerator
from app.ai.memory.conversation_memory import ConversationMemory
from app.ai.retrieval.base import BaseRetriever
from app.core.config import settings
from app.workflows.state import ChatState


class ChatWorkflow:
    """
    Builds and runs the chat LangGraph.

    The compiled graph is created once per instance and reused across calls.
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
        self._citation_builder = citation_builder
        self._memory = memory
        self._graph = self._build()

    # ---- nodes -------------------------------------------------------

    def _load_history(self, state: ChatState) -> ChatState:
        conversation_id = state.get("conversation_id")
        history = self._memory.get_history(conversation_id) if conversation_id else None
        return {"history": history}

    async def _retrieve(self, state: ChatState) -> ChatState:
        documents = await self._retriever.retrieve(
            query=state["question"],
            k=state.get("k") or settings.RETRIEVAL_TOP_K,
        )
        return {"documents": documents}

    async def _generate(self, state: ChatState) -> ChatState:
        answer = await self._generator.generate(
            question=state["question"],
            documents=state.get("documents", []),
            history=state.get("history"),
        )
        return {"answer": answer}

    def _cite(self, state: ChatState) -> ChatState:
        citations = self._citation_builder.build(state.get("documents", []))
        conversation_id = state.get("conversation_id")
        if conversation_id:
            self._memory.add_turn(
                conversation_id, state["question"], state.get("answer", "")
            )
        return {"citations": citations}

    # ---- graph -------------------------------------------------------

    def _build(self):
        graph = StateGraph(ChatState)
        graph.add_node("load_history", self._load_history)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("cite", self._cite)

        graph.add_edge(START, "load_history")
        graph.add_edge("load_history", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "cite")
        graph.add_edge("cite", END)
        return graph.compile()

    async def run(
        self,
        question: str,
        conversation_id: str | None = None,
        k: int | None = None,
    ) -> ChatState:
        """Execute the graph and return the final ChatState."""
        return await self._graph.ainvoke(
            {
                "question": question,
                "conversation_id": conversation_id,
                "k": k or settings.RETRIEVAL_TOP_K,
            }
        )
