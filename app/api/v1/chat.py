"""
chat.py

Conversational RAG API endpoints.

``POST /chat``         — returns a complete answer plus citations.
``POST /chat/stream``  — streams the answer as plain-text chunks.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.ai.generation.citation_builder import Citation
from app.core.dependencies import get_chat_service
from app.domain.chat.chat_service import ChatService

router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="The user's question.")
    conversation_id: str | None = Field(
        default=None, description="Optional id enabling multi-turn memory."
    )
    k: int | None = Field(
        default=None, ge=1, le=50, description="Max chunks to retrieve as context."
    )


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    conversation_id: str | None = None


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    """Answer a question with citations, grounded in the knowledge base."""
    result = await chat_service.chat(
        question=request.question,
        conversation_id=request.conversation_id,
        k=request.k,
    )
    return ChatResponse(
        answer=result.answer,
        citations=result.citations,
        conversation_id=result.conversation_id,
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    """Stream the answer token-by-token as ``text/plain``."""
    token_stream = chat_service.stream_chat(
        question=request.question,
        conversation_id=request.conversation_id,
        k=request.k,
    )
    return StreamingResponse(token_stream, media_type="text/plain")
