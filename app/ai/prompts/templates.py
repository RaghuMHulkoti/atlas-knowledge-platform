"""
templates.py

Prompt templates for Atlas retrieval-augmented generation.

Kept free of any provider or retrieval details so prompts can be tuned in one
place without touching the generation code.
"""

from langchain_core.documents import Document

# System instruction that governs how the assistant answers.
RAG_SYSTEM_PROMPT = (
    "You are Atlas, an engineering knowledge assistant. "
    "Answer the user's question using ONLY the numbered context sources below. "
    "Cite every claim with its source number in square brackets, e.g. [1] or [2]. "
    "If the context does not contain the answer, say so plainly and do not "
    "invent facts. Be concise and technical."
)

# Assembled into the final user turn.
RAG_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Context sources:\n{context}\n\n"
    "Answer the question using the sources above and cite them by number."
)

# Returned when retrieval yields nothing to ground the answer.
NO_CONTEXT_MESSAGE = (
    "I couldn't find anything relevant in the indexed knowledge base to answer "
    "that. Try ingesting the relevant repository or rephrasing your question."
)


def format_context(documents: list[Document]) -> str:
    """
    Render retrieved chunks into a numbered context block.

    Each source is numbered starting at 1 so the model can cite it as [n].
    The numbering here matches the citation indices produced by CitationBuilder.
    """
    blocks: list[str] = []
    for index, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        label = meta.get("path") or meta.get("title") or meta.get("source") or "source"
        blocks.append(f"[{index}] ({label})\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_rag_user_prompt(question: str, documents: list[Document]) -> str:
    """Build the final user-turn prompt from a question and retrieved chunks."""
    return RAG_USER_TEMPLATE.format(
        question=question,
        context=format_context(documents),
    )
