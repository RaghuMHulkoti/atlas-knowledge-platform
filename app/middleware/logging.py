"""
logging.py

Structured access logging.

Emits one log line per request with method, path, status code, and elapsed
time, prefixed with the current request id. Uses ``time.perf_counter`` for
duration so it does not depend on the (unavailable-in-some-sandboxes) wall clock
for timing.
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.logging import get_logger
from app.middleware.request_id import get_request_id
from app.observability.metrics import metrics

logger = get_logger("atlas.access")


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Log every request's method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "[%s] %s %s -> ERROR (%.1fms)",
                get_request_id(),
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics.increment("http.requests.total")
        metrics.increment(f"http.responses.{response.status_code // 100}xx")
        logger.info(
            "[%s] %s %s -> %d (%.1fms)",
            get_request_id(),
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
