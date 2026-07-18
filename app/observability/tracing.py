"""
tracing.py

Minimal span timing.

A context manager that measures how long a named block takes and logs it. Uses
``time.perf_counter`` so it works where the wall clock is unavailable. Swap the
body for an OpenTelemetry span when distributed tracing is introduced — the call
site (``with span("name"):``) stays the same.
"""

import time
from contextlib import contextmanager

from app.core.logging import get_logger
from app.observability.metrics import metrics

logger = get_logger("atlas.trace")


@contextmanager
def span(name: str):
    """
    Time the wrapped block, log its duration, and record it as a metric.

    Example::

        with span("indexing"):
            pipeline.run(documents)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics.increment(f"span.{name}.count")
        metrics.increment(f"span.{name}.total_ms", elapsed_ms)
        logger.debug("span %s took %.1fms", name, elapsed_ms)
