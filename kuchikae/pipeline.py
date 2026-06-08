"""Kuchikae pipeline orchestration."""

from __future__ import annotations

import time

from kuchikae.audio_cache import AudioCache
from kuchikae.stt import DummySTTBackend, STTBackend
from kuchikae.text_transform import DummyTextTransformBackend, TextTransformBackend
from kuchikae.types import (
    LatencyReport,
    PipelineResult,
    TextTransformPrompt,
    VoiceOutputPrompt,
)
from kuchikae.voice_context import VoiceContextExtractor
from kuchikae.voice_output import DummyVoiceOutputBackend, VoiceOutputBackend


class KuchikaePipeline:
    """Run the prompt-conditioned, voice-conditioned scaffold pipeline."""

    def __init__(
        self,
        audio_cache: AudioCache | None = None,
        voice_context_extractor: VoiceContextExtractor | None = None,
        stt_backend: STTBackend | None = None,
        text_transform_backend: TextTransformBackend | None = None,
        voice_output_backend: VoiceOutputBackend | None = None,
    ) -> None:
        self.audio_cache = audio_cache or AudioCache()
        self.voice_context_extractor = voice_context_extractor or VoiceContextExtractor()
        self.stt_backend = stt_backend or DummySTTBackend()
        self.text_transform_backend = text_transform_backend or DummyTextTransformBackend()
        self.voice_output_backend = voice_output_backend or DummyVoiceOutputBackend()

    def process(
        self,
        audio_path: str,
        text_transform_prompt: TextTransformPrompt,
        voice_output_prompt: VoiceOutputPrompt,
    ) -> PipelineResult:
        """Process one utterance through the v0.1 dummy pipeline."""
        total_start = time.perf_counter()

        self.audio_cache.add_utterance(audio_path)

        reference_audio_path = self.audio_cache.get_reference_audio_path()
        voice_context = self.voice_context_extractor.extract(reference_audio_path)

        stt_start = time.perf_counter()
        source_text = self.stt_backend.transcribe(audio_path)
        stt_seconds = time.perf_counter() - stt_start

        text_transform_start = time.perf_counter()
        transformed_text = self.text_transform_backend.transform(
            source_text,
            text_transform_prompt,
        )
        text_transform_seconds = time.perf_counter() - text_transform_start

        voice_output_start = time.perf_counter()
        output_audio_path = self.voice_output_backend.synthesize(
            text=transformed_text,
            voice_context=voice_context,
            prompt=voice_output_prompt,
        )
        voice_output_seconds = time.perf_counter() - voice_output_start

        total_seconds = time.perf_counter() - total_start

        return PipelineResult(
            source_text=source_text,
            transformed_text=transformed_text,
            output_audio_path=output_audio_path,
            text_transform_prompt=text_transform_prompt.instruction,
            voice_output_prompt=voice_output_prompt.instruction,
            voice_ready=voice_context.ready,
            latency=LatencyReport(
                stt_seconds=stt_seconds,
                text_transform_seconds=text_transform_seconds,
                voice_output_seconds=voice_output_seconds,
                total_seconds=total_seconds,
            ),
        )
