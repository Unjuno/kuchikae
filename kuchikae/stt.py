"""STTBackend and DummySTTBackend."""

from __future__ import annotations


class STTBackend:
    """Abstract base for speech-to-text backends."""

    def transcribe(self, audio_path: str) -> str:  # pragma: no cover
        raise NotImplementedError


class DummySTTBackend(STTBackend):
    """Returns a fixed Japanese sentence for v0.1."""

    def transcribe(self, audio_path: str) -> str:
        return "明日までに資料を送って"
