"""Audio cache for utterance and reference audio paths."""

from __future__ import annotations


class AudioCache:
    """Store the latest utterance and reference audio path.

    v0.1 uses the same file as both the latest utterance and the reference audio.
    Later versions may replace this with a rolling reference window.
    """

    def __init__(self) -> None:
        self._latest_utterance_path: str | None = None
        self._reference_audio_path: str | None = None

    def add_utterance(self, audio_path: str) -> None:
        """Add an utterance and make it the active reference audio."""
        self._latest_utterance_path = audio_path
        self._reference_audio_path = audio_path

    def get_latest_utterance_path(self) -> str | None:
        """Return the latest utterance audio path."""
        return self._latest_utterance_path

    def get_reference_audio_path(self) -> str | None:
        """Return the active reference audio path."""
        return self._reference_audio_path
