"""
request_id.py

Assigns a unique id to every request.

The id is taken from the inbound ``X-Request-ID`` header when present, otherwise
generated. It is stored in a context variable so any log line produced while
handling the request can include it, and echoed back on the response.
"""

import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-ID"

# Readable by loggers anywhere in the request lifecycle.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request id to the context and the response headers."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def get_request_id() -> str:
    """Return the current request id, or ``"-"`` outside a request."""
    return request_id_ctx.get()
