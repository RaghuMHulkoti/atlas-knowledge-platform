from fastapi import Request
from fastapi.responses import JSONResponse


class AtlasException(Exception):
    """Base exception for Atlas."""


class ConnectorException(AtlasException):
    """Raised when a connector fails."""


class RetrievalException(AtlasException):
    """Raised during retrieval failures."""


class LLMException(AtlasException):
    """Raised when the LLM call fails."""


class IndexingException(AtlasException):
    """Raised when document chunking, embedding, or vector store writes fail."""


async def atlas_exception_handler(
    request: Request,
    exc: AtlasException,
):
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "message": str(exc),
        },
    )
