"""Processing cache for performance optimization."""

from __future__ import annotations

import logging
from typing import Dict

from kuchikae.domain.audio_key import AudioKey
from kuchikae.domain.types import PipelineResult, TextTransformPrompt, VoiceContext

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
        self._text_cache: Dict[tuple[str, str], str] = {}
        self._voice_output_cache: Dict[tuple[str, str, str], str] = {}
        self._voice_context_cache: Dict[AudioKey, VoiceContext] = {}
        self._result_cache: Dict[tuple[AudioKey, str, str], PipelineResult] = {}

    def get_stt(self, audio_key: AudioKey) -> str | None:
        """Get STT transcript from cache."""
        return self._stt_cache.get(audio_key)

    def set_stt(self, audio_key: AudioKey, transcript: str) -> None:
        """Set STT transcript in cache."""
        self._stt_cache[audio_key] = transcript
        logger.debug("cached STT for %s", audio_key)

    def get_text(self, source_text: str, prompt: TextTransformPrompt) -> str | None:
        return self._text_cache.get((source_text, prompt.instruction))

    def set_text(self, source_text: str, prompt: TextTransformPrompt, transformed_text: str) -> None:
        self._text_cache[(source_text, prompt.instruction)] = transformed_text
        logger.debug("cached transformed text for prompt_len=%d", len(prompt.instruction))

    def get_voice_context(self, audio_key: AudioKey) -> VoiceContext | None:
        """Get voice context from cache."""
        return self._voice_context_cache.get(audio_key)

    def set_voice_context(self, audio_key: AudioKey, voice_context: VoiceContext) -> None:
        """Set voice context in cache."""
        self._voice_context_cache[audio_key] = voice_context
        logger.debug("cached voice context for %s", audio_key)

    def get_voice_output(self, transformed_text: str, voice_prompt: str, voice_context: VoiceContext) -> str | None:
        cache_key = (transformed_text, voice_prompt, voice_context.reference_audio_path or "")
        return self._voice_output_cache.get(cache_key)

    def set_voice_output(self, transformed_text: str, voice_prompt: str, voice_context: VoiceContext, output_path: str) -> None:
        cache_key = (transformed_text, voice_prompt, voice_context.reference_audio_path or "")
        self._voice_output_cache[cache_key] = output_path
        logger.debug("cached voice output for prompt_len=%d", len(voice_prompt))

    def get_result(self, audio_key: AudioKey, text_prompt: str, voice_prompt: str) -> PipelineResult | None:
        return self._result_cache.get((audio_key, text_prompt, voice_prompt))

    def set_result(self, audio_key: AudioKey, text_prompt: str, voice_prompt: str, result: PipelineResult) -> None:
        self._result_cache[(audio_key, text_prompt, voice_prompt)] = result
        logger.debug(
            "cached full result for prompt_len=%d voice_prompt_len=%d",
            len(text_prompt),
            len(voice_prompt),
        )

    def clear(self) -> None:
        """Clear all cached items."""
        self._stt_cache.clear()
        self._text_cache.clear()
        self._voice_output_cache.clear()
        self._voice_context_cache.clear()
        self._result_cache.clear()

    def __len__(self) -> int:
        return len(self._stt_cache) + len(self._voice_context_cache)

    def __repr__(self) -> str:
        return (
            f"ProcessingCache(stt_entries={len(self._stt_cache)}, "
            f"text_entries={len(self._text_cache)}, "
            f"voice_output_entries={len(self._voice_output_cache)}, "
            f"voice_context_entries={len(self._voice_context_cache)}, "
            f"result_entries={len(self._result_cache)})"
        )
