"""Dummy voice context extractor."""

from __future__ import annotations

from kuchikae.types import VoiceContext


class DummyVoiceContextExtractor:
    """Dummy implementation of VoiceContextExtractor."""

    def extract(self, reference_audio_path: str | None) -> VoiceContext:
        """Extract VoiceContext from reference audio.

        v0.1 dummy behavior:
        - If a reference path exists or is provided, return VoiceContext(..., ready=True).
        - Do not compute real speaker embeddings.
        - Do not require a model.
        """
        if reference_audio_path is None or not reference_audio_path:
            return VoiceContext(reference_audio_path="", ready=False)

        return VoiceContext(reference_audio_path=reference_audio_path, ready=True)
