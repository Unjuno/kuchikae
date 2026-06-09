"""Audio cache with rolling reference window for multi-frame SE extraction."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceAudio:
    """A single utterance with quality metadata."""

    audio_path: str
    quality_score: float = 1.0  # higher = better voice quality


class AudioCache:
    """Stores recent utterances for multi-frame SE extraction.

    Maintains a rolling window of the best N references; uses the highest-
    scored one as the primary reference tone color source.
    """

    def __init__(self, max_references: int = 5) -> None:
        self._window: list[ReferenceAudio] = []
        self.max_references = max_references

    @property
    def count(self) -> int:
        return len(self._window)

    def add_utterance(
        self, audio_path: str, quality_score: float | None = None,
    ) -> None:
        """Add a new utterance to the rolling window."""
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        ref = ReferenceAudio(
            audio_path=audio_path,
            quality_score=quality_score or 1.0,
        )

        self._window.append(ref)
        # Trim oldest entries if window is full.
        while len(self._window) > self.max_references:
            self._window.pop(0)

    def get_best_reference(self) -> str | None:
        """Return the path with highest quality_score."""
        if not self._window:
            return None
        best = max(self._window, key=lambda r: r.quality_score)
        return best.audio_path

    def get_latest_utterance_path(self) -> str | None:
        """Return the most recently added utterance path."""
        if not self._window:
            return None
        return self._window[-1].audio_path

    def get_reference_audio_path(self) -> str | None:
        """Alias for get_best_reference (backward compatibility)."""
        return self.get_best_reference()
