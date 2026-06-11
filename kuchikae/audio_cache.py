"""AudioCache and VoiceContextExtractor classes as per ARCHITECTURE.md."""

from __future__ import annotations

import logging
import os

from kuchikae.types import AudioSegment, VoiceContext

logger = logging.getLogger(__name__)


class AudioCache:
    """AudioCache stores paths, not model features."""

    def __init__(self) -> None:
        self._utterances: dict[str, str] = {}
        self._references: dict[str, str] = {}

    def add_utterance(self, audio_path: str) -> None:
        """Add utterance audio path to cache."""
        self._utterances[audio_path] = audio_path
        logger.debug("added utterance: %s", audio_path)

    def get_latest_utterance_path(self) -> str | None:
        """Get the most recent utterance path."""
        if not self._utterances:
            return None
        return next(iter(reversed(list(self._utterances.keys()))))

    def get_reference_audio_path(self) -> str | None:
        """Get reference audio path if available."""
        return next(iter(self._references.values()), None)


class VoiceContextExtractor:
    """Extract VoiceContext from reference audio."""

    def extract(self, reference_audio_path: str | None) -> VoiceContext:
        """Extract VoiceContext from reference audio.

        v0.1 dummy behavior:
        - If a reference path exists or is provided, return VoiceContext(..., ready=True).
        - Do not compute real speaker embeddings.
        - Do not require a model.
        """
        if reference_audio_path is None or not os.path.isfile(reference_audio_path):
            return VoiceContext(reference_audio_path="", ready=False)

        return VoiceContext(reference_audio_path=reference_audio_path, ready=True)
