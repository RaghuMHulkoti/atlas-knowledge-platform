"""
metrics.py

Lightweight in-process metrics.

A dependency-free counter/gauge registry — enough to observe request volume and
pipeline activity without pulling in Prometheus. For a multi-process deployment,
replace the registry with a Prometheus client; callers use the same API.
"""

from collections import defaultdict
from threading import Lock


class MetricsRegistry:
    """Thread-safe counters and gauges keyed by name."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}

    def increment(self, name: str, value: float = 1.0) -> None:
        """Add *value* to the named counter."""
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """Set the named gauge to *value*."""
        with self._lock:
            self._gauges[name] = value

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Return a copy of all current counters and gauges."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
            }


# Process-wide singleton.
metrics = MetricsRegistry()
