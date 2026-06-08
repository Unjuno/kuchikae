"""Timing utilities for Kuchikae."""

from __future__ import annotations

import time


class Timer:
    """Simple context-manager timer."""

    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self

    @property
    def elapsed(self) -> float:
        return time.time() - self._start


def now_ms() -> int:
    return int(time.time() * 1000)
