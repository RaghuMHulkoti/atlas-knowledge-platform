"""
exception.py

Catch-all exception handler.

Ensures any exception not already handled by a more specific handler is returned
as a uniform JSON 500 (never a raw stack trace to the client) and is logged with
the current request id for correlation.
"""

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.middleware.request_id import get_request_id

logger = get_logger("atlas.error")


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log an unexpected exception and return a generic 500 response."""
    request_id = get_request_id()
    logger.exception(
        "[%s] Unhandled exception on %s %s",
        request_id,
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred.",
            "request_id": request_id,
        },
    )
