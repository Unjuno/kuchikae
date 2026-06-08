"""KuchikaePipeline — the main processing pipeline."""

from __future__ import annotations

import time

from kuchikae.audio_cache import AudioCache
from kuchikae.stt import DummySTTBackend, STTBackend
from kuchikae.text_transform import (
    DummyTextTransformBackend,
    TextTransformBackend,
)
from kuchikae.types import (
    LatencyReport,
    PipelineResult,
    TextTransformPrompt,
    VoiceContext,
    VoiceOutputPrompt,
)
from kuchikae.voice_context import VoiceContextExtractor
from kuchikae.voice_output import DummyVoiceOutputBackend, VoiceOutputBackend


class KuchikaePipeline:
    """Run the full voice-transform pipeline."""

    def __init__(
        self,
        stt_backend: STTBackend | None = None,
        text_transform_backend: TextTransformBackend | None = None,
        voice_output_backend: VoiceOutputBackend | None = None,
    ) -> None:
        self.stt_backend = stt_backend or DummySTTBackend()
        self.text_transform_backend = text_transform_backend or DummyTextTransformBackend()
        self.voice_output_backend = voice_output_backend or DummyVoiceOutputBackend()

    def process(
        self,
        audio_path: str,
        text_transform_prompt: TextTransformPrompt,
        voice_output_prompt: VoiceOutputPrompt,
    ) -> PipelineResult:
        cache = AudioCache()
        extractor = VoiceContextExtractor()

        total_start = time.time()

        # 1. Add utterance to AudioCache
        cache.add_utterance(audio_path)

        # 2. Extract voice context from reference audio
        voice_context = extractor.extract(cache)

        # 3. Transcribe
        stt_start = time.time()
        source_text = self.stt_backend.transcribe(audio_path)
        stt_seconds = time.time() - stt_start

        # 4. Transform text with prompt
        tx_start = time.time()
        transformed_text = self.text_transform_backend.transform(source_text, text_transform_prompt)
        text_transform_seconds = time.time() - tx_start

        # 5. Synthesize output audio
        vo_start = time.time()
        output_audio_path = self.voice_output_backend.synthesize(
            text=transformed_text,
            voice_context=voice_context,
            prompt=voice_output_prompt,
        )
        voice_output_seconds = time.time() - vo_start

        total_seconds = time.time() - total_start

        latency = LatencyReport(
            stt_seconds=stt_seconds,
            text_transform_seconds=text_transform_seconds,
            voice_output_seconds=voice_output_seconds,
            total_seconds=total_seconds,
        )

        return PipelineResult(
            source_text=source_text,
            transformed_text=transformed_text,
            output_audio_path=output_audio_path,
            voice_context=voice_context,
            latency=latency,
        )
