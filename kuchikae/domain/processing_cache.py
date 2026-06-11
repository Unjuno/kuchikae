"""Processing cache for performance optimization."""

from __future__ import annotations

import logging
from typing import Any, Dict

from kuchikae.domain.audio_key import AudioKey
from kuchikae.domain.types import TextTransformPrompt, VoiceContext

logger = logging.getLogger(__name__)


class ProcessingCache:
    """Processing cache for performance optimization.

    transcript cache by audio key
    voice context cache by audio key
    segment cache by audio key, if segmenting scaffold is added

    Do not cache text transform by text only.
    If text transform caching is added, key must include:
    - source text
    - TextTransformPrompt fields

    Do not cache voice output by text only.
    If voice output caching is added, key must include:
    - transformed text
    - VoiceOutputPrompt fields
    - voice context key
    """

    def __init__(self) -> None:
        self._stt_cache: Dict[AudioKey, str] = {}
        self._voice_context_cache: Dict[AudioKey, VoiceContext] = {}

    def get_stt(self, audio_key: AudioKey) -> str | None:
        """Get STT transcript from cache."""
        return self._stt_cache.get(audio_key)

    def set_stt(self, audio_key: AudioKey, transcript: str) -> None:
        """Set STT transcript in cache."""
        self._stt_cache[audio_key] = transcript
        logger.debug("cached STT for %s", audio_key)

    def get_voice_context(self, audio_key: AudioKey) -> VoiceContext | None:
        """Get voice context from cache."""
        return self._voice_context_cache.get(audio_key)

    def set_voice_context(self, audio_key: AudioKey, voice_context: VoiceContext) -> None:
        """Set voice context in cache."""
        self._voice_context_cache[audio_key] = voice_context
        logger.debug("cached voice context for %s", audio_key)

    def clear(self) -> None:
        """Clear all cached items."""
        self._stt_cache.clear()
        self._voice_context_cache.clear()

    def __len__(self) -> int:
        return len(self._stt_cache) + len(self._voice_context_cache)

    def __repr__(self) -> str:
        return (
            f"ProcessingCache(stt_entries={len(self._stt_cache)}, "
            f"voice_context_entries={len(self._voice_context_cache)})"
        )
