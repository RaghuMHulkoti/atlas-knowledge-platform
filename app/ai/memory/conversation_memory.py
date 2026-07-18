"""
conversation_memory.py

In-process conversation history for multi-turn chat.

A minimal, thread-unsafe, in-memory store keyed by conversation id. Sufficient
for a single-process deployment; swap for Redis/DB-backed storage when the
platform scales horizontally. History is trimmed to a bounded window so prompts
stay within model context limits.
"""

from collections import defaultdict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ConversationMemory:
    """
    Stores recent (human, ai) turns per conversation id.

    Responsibilities:
    - Append user and assistant turns.
    - Return a bounded, ordered history window for prompting.

    Non-responsibilities:
    - Persistence across process restarts.
    - Summarisation or token-aware trimming.
    """

    def __init__(self, max_messages: int = 10) -> None:
        # max_messages counts individual messages (human + ai), most recent kept.
        self._max_messages = max_messages
        self._store: dict[str, list[BaseMessage]] = defaultdict(list)

    def get_history(self, conversation_id: str) -> list[BaseMessage]:
        """Return the bounded message history for *conversation_id*."""
        return list(self._store.get(conversation_id, []))

    def add_turn(
        self,
        conversation_id: str,
        question: str,
        answer: str,
    ) -> None:
        """Append a completed (question, answer) turn and trim to the window."""
        history = self._store[conversation_id]
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        # Keep only the most recent messages.
        if len(history) > self._max_messages:
            del history[: len(history) - self._max_messages]

    def clear(self, conversation_id: str) -> None:
        """Forget all history for *conversation_id*."""
        self._store.pop(conversation_id, None)
