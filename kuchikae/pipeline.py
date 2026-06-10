"""Kuchikae pipeline orchestration."""

from __future__ import annotations

import os
import time

from kuchikae.audio_cache import AudioCache
from kuchikae.stt import (
    DummySTTBackend,
    FasterWhisperSTTBackend,
    STTBackend,
)
from kuchikae.text_transform import (
    DummyTextTransformBackend,
    GPTTextTransformBackend,
    RuleTextTransformBackend,
    TextTransformBackend,
)
from kuchikae.types import (
    LatencyReport,
    PipelineResult,
    TextTransformPrompt,
    VoiceOutputPrompt,
)
from kuchikae.voice_context import VoiceContextExtractor
from kuchikae.voice_output import (
    DummyVoiceOutputBackend,
    OpenVoiceOutputBackend,
    VoiceOutputBackend,
)


def create_pipeline(backend_config: dict | None = None) -> KuchikaePipeline:
    """Factory to create a pipeline with real or dummy backends based on config.

    Config keys (all optional):
        stt_backend: "dummy" | "faster_whisper" (default: auto-detect faster-whisper).
        text_transform_backend: "dummy" | "rule" | "gpt_oss" (default: "rule").
        voice_output_backend: "dummy" | "openvoice" (default: "dummy" unless OPENVOICE_READY=1).

    Example:
        >>> pipeline = create_pipeline({"text_transform_backend": "gpt_oss"})
    """
    config = backend_config or {}

    if config.get("stt_backend") == "faster_whisper":
        stt = FasterWhisperSTTBackend()
    else:
        # Auto-detect faster-whisper.
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            has_faster_whisper = True
        except ImportError:
            has_faster_whisper = False

        stt = FasterWhisperSTTBackend() if has_faster_whisper else DummySTTBackend()

    text_backend_type = config.get("text_transform_backend", "rule")
    text_backends = {"rule": RuleTextTransformBackend, "gpt_oss": GPTTextTransformBackend}
    tt_class = text_backends.get(text_backend_type, RuleTextTransformBackend)
    if tt_class == GPTTextTransformBackend and not os.environ.get("OPENAI_API_KEY"):
        tt_class = DummyTextTransformBackend  # fall back gracefully.

    voice_output_type = config.get("voice_output_backend", "dummy")
    if voice_output_type == "openvoice" or (config.get("auto_openvoice") and os.environ.get("OPENVOICE_READY")):
        vo = OpenVoiceOutputBackend()
    else:
        # Auto-detect OpenVoice.
        _ow_ready = os.path.isdir("/Users/taka/repos/OpenVoice") and os.environ.get("OPENVOICE_READY")
        vo = OpenVoiceOutputBackend() if _ow_ready else DummyVoiceOutputBackend()

    return KuchikaePipeline(
        stt_backend=stt,
        text_transform_backend=tt_class(),
        voice_output_backend=vo,
    )


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
