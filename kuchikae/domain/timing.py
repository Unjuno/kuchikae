"""Timing utilities for Kuchikae."""

from __future__ import annotations

import time


class Timer:
    """Simple context-manager timer using ``time.time``."""

    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self

    @property
    def elapsed(self) -> float:
        return time.time() - self._start


class PerfTimer:
    """High-resolution context-manager timer using ``time.perf_counter``.

    Use for micro-benchmarks where sub-millisecond precision matters.
    """

    def __enter__(self) -> "PerfTimer":
        self._start = time.perf_counter()
        return self

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start


def now_ms() -> int:
    return int(time.time() * 1000)
