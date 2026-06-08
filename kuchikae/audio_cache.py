"""AudioCache — buffers utterances and tracks reference audio."""

from __future__ import annotations


class AudioCache:
    """Holds a list of utterance paths plus the active reference path."""

    def __init__(self) -> None:
        self._utterances: list[str] = []
        self._reference_path: str | None = None
        self._latest_utterance_path: str | None = None

    @property
    def utterances(self) -> list[str]:
        return list(self._utterances)

    @property
    def reference_path(self) -> str | None:
        return self._reference_path

    @property
    def latest_utterance_path(self) -> str | None:
        return self._latest_utterance_path

    def add_utterance(self, path: str) -> None:
        """Append an utterance path and mark it as the latest."""
        self._utterances.append(path)
        self._latest_utterance_path = path

    def set_reference_audio(self, path: str) -> None:
        """Set the reference audio used for voice context extraction."""
        self._reference_path = path
