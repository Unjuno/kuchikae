"""Kuchikae pipeline orchestration."""

from __future__ import annotations

import os
import time
from typing import Generator

from kuchikae.domain.audio import AudioSegmenter, FixedWindowSegmenter
from kuchikae.domain.audio_cache import AudioCache, VoiceContextExtractor, DummyVoiceContextExtractor
from kuchikae.domain.audio_key import AudioKey, AudioKeyFromCacheKey
from kuchikae.domain.metrics import LatencyLogger
from kuchikae.counting_backends import (
    CountingSTTBackend,
    CountingTextTransformBackend,
    CountingVoiceContextExtractor,
    CountingVoiceOutputBackend,
)
from kuchikae.domain.processing_cache import ProcessingCache
from kuchikae.domain.stt import (
    DummySTTBackend,
    FasterWhisperSTTBackend,
    SegmentedSTTBackend,
    StreamingFasterWhisperSTTBackend,
    STTBackend,
)
from kuchikae.domain.text_transform import (
    DummyTextTransformBackend,
    GPTTextTransformBackend,
    OllamaTextTransformBackend,
    PromptedRuleTextTransformBackend,
    RuleTextTransformBackend,
    TextTransformBackend,
)
from kuchikae.domain.types import (
    AudioCacheKey,
    PipelineResult,
    StreamingLatencyReport,
    TextTransformPrompt,
)
from kuchikae.domain.voice_output import (
    DummyVoiceOutputBackend,
    IrodoriTTSVoiceOutputBackend,
    OpenVoiceOutputBackend,
    VoiceOutputBackend,
)
from kuchikae.logging import setup_logger
from kuchikae.pipeline.audio_validation import validate_audio

logger = setup_logger("kuchikae.pipeline")


def create_pipeline(backend_config: dict | None = None) -> KuchikaePipeline:
    config = backend_config or {}

    stt_type = config.get("stt_backend", "faster_whisper")
    use_segmented = config.get("segmented_stt", False)
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        has_faster_whisper = True
    except ImportError:
        has_faster_whisper = False

    if stt_type == "faster_whisper" and has_faster_whisper:
        inner = FasterWhisperSTTBackend()
    else:
        inner = DummySTTBackend()

    stt: STTBackend
    if use_segmented:
        segmenter: AudioSegmenter = FixedWindowSegmenter(chunk_sec=30.0, overlap_sec=2.0)
        stt = SegmentedSTTBackend(inner=inner, segmenter=segmenter)
    elif config.get("streaming_stt", False):
        stt = StreamingFasterWhisperSTTBackend(chunk_sec=5.0, overlap_sec=1.0)
    else:
        stt = inner

    text_backend_type = config.get("text_transform_backend", "prompted_rule")
    text_model = config.get("text_transform_model")
    text_backends = {
        "ollama": OllamaTextTransformBackend,
        "rule": RuleTextTransformBackend,
        "prompted_rule": PromptedRuleTextTransformBackend,
        "gpt_oss": GPTTextTransformBackend,
    }
    tt_class = text_backends.get(text_backend_type, OllamaTextTransformBackend)
    tt_kwargs = {}
    if text_model:
        tt_kwargs["model"] = text_model
    if tt_class == GPTTextTransformBackend and not os.environ.get("OPENAI_API_KEY"):
        tt_class = DummyTextTransformBackend

    voice_output_type = config.get("voice_output_backend", "irodori")
    _ow_path = "/Users/taka/repos/OpenVoice"
    _irodori_ready = False
    try:
        from irodori_tts.inference_runtime import InferenceRuntime  # noqa: F401
        _irodori_ready = True
    except ImportError:
        pass

    if voice_output_type == "irodori" and _irodori_ready:
        vo = IrodoriTTSVoiceOutputBackend()
    elif voice_output_type == "openvoice" or (config.get("auto_openvoice") and os.environ.get("OPENVOICE_READY")):
        vo = OpenVoiceOutputBackend()
    else:
        _ow_ready = os.path.isdir(_ow_path) and os.environ.get("OPENVOICE_READY")
        vo = OpenVoiceOutputBackend() if _ow_ready else DummyVoiceOutputBackend()

    logger.info("pipeline: stt=%s text=%s(%s) voice=%s",
                type(stt).__name__, type(tt_class()).__name__,
                text_model or "default", type(vo).__name__)

    return KuchikaePipeline(
        stt_backend=stt,
        text_transform_backend=tt_class(**tt_kwargs),
        voice_output_backend=vo,
    )


class KuchikaePipeline:

    def __init__(
        self,
        stt_backend: STTBackend | None = None,
        text_transform_backend: TextTransformBackend | None = None,
        voice_output_backend: VoiceOutputBackend | None = None,
        processing_cache: ProcessingCache | None = None,
        latency_logger: LatencyLogger | None = None,
    ) -> None:
        self.stt_backend = stt_backend or DummySTTBackend()
        self.text_transform_backend = text_transform_backend or DummyTextTransformBackend()
        self.voice_output_backend = voice_output_backend or DummyVoiceOutputBackend()
        self.processing_cache = processing_cache or ProcessingCache()
        self.latency_logger = latency_logger
        self._audio_cache = AudioCache()
        self._voice_context_extractor = VoiceContextExtractor()

    def warmup(self) -> None:
        logger.info("warming up...")

        if isinstance(self.stt_backend, FasterWhisperSTTBackend):
            try:
                logger.info("warming up STT...")
                self.stt_backend.transcribe.__self__.transcribe("")  # noqa: SLF001
            except Exception as e:
                logger.debug("STT warmup skipped: %s", e)

        if hasattr(self.voice_output_backend, "_ensure_runtime"):
            try:
                logger.info("warming up voice model...")
                self.voice_output_backend._ensure_runtime()  # noqa: SLF001
            except Exception as e:
                logger.debug("voice warmup skipped: %s", e)

        if hasattr(self.text_transform_backend, "model"):
            try:
                logger.info("warming up LLM...")
                import httpx
                httpx.post(
                    f"{self.text_transform_backend._base_url}/api/generate",
                    json={
                        "model": self.text_transform_backend.model,
                        "prompt": "warmup",
                        "stream": False,
                        "keep_alive": "5m",
                    },
                    timeout=10,
                )
            except Exception as e:
                logger.debug("LLM warmup skipped: %s", e)

        logger.info("warmup done")

    def check_audio(self, audio_path: str) -> None:
        validate_audio(audio_path)

    def _step_stt(self, audio_path: str, audio_key: AudioKey) -> str:
        cached = self.processing_cache.get_stt(audio_key)
        if cached is not None:
            return cached
        text = self.stt_backend.transcribe(audio_path)
        self.processing_cache.set_stt(audio_key, text)
        return text

    def _step_voice(self, text: str, audio_path: str, audio_key: AudioKey, voice_context: VoiceContext) -> str:
        cached_context = self.processing_cache.get_voice_context(audio_key)
        if cached_context is not None:
            voice_context = cached_context
        else:
            self.processing_cache.set_voice_context(audio_key, voice_context)
        
        out = self.voice_output_backend.synthesize(text, voice_context)
        return out

    def process(
        self,
        audio_path: str,
        text_transform_prompt: TextTransformPrompt,
    ) -> PipelineResult:
        t0 = time.time()

        self.check_audio(audio_path)
        cache_key = AudioCacheKey.from_file(audio_path)
        audio_key = AudioKeyFromCacheKey(cache_key)

        t1 = time.time()
        source_text = self._step_stt(audio_path, audio_key)
        stt_latency = time.time() - t1
        logger.info("STT: %.2fs → %s", stt_latency, source_text[:60])

        t2 = time.time()
        transformed_text = self.text_transform_backend.transform(source_text, text_transform_prompt)
        text_latency = time.time() - t2
        logger.info("TEXT: %.2fs → %s", text_latency, transformed_text[:60])

        t3 = time.time()
        voice_context = self._voice_context_extractor.extract(audio_path)
        text_for_voice = transformed_text.strip() or source_text
        output_audio_path = self._step_voice(text_for_voice, audio_path, audio_key, voice_context)
        voice_latency = time.time() - t3
        logger.info("VOICE: %.2fs → %s", voice_latency, output_audio_path)

        total = time.time() - t0
        logger.info("TOTAL: %.2fs (STT %.2f + TEXT %.2f + VOICE %.2f)",
                    total, stt_latency, text_latency, voice_latency)

        result = PipelineResult(
            output_audio_path=output_audio_path,
            source_text=source_text,
            transformed_text=transformed_text,
            stt_latency=stt_latency,
            text_transform_latency=text_latency,
            voice_output_latency=voice_latency,
            total_latency=total,
        )

        if self.latency_logger is not None:
            report = StreamingLatencyReport(
                session_id=cache_key.path,
                recording_started_at=None,
                processing_finished_at=time.time(),
            )
            self.latency_logger.log_report(report)

        return result

    def process_stream(
        self,
        audio_path: str,
        text_transform_prompt: TextTransformPrompt,
    ) -> Generator[tuple[str, str | None, str | None, str | None], None, None]:
        t0 = time.time()
        self.check_audio(audio_path)
        cache_key = AudioCacheKey.from_file(audio_path)
        audio_key = AudioKeyFromCacheKey(cache_key)

        yield "STT", None, None, None
        source_text = self._step_stt(audio_path, audio_key)
        logger.info("STT: %.2fs %s", time.time() - t0, source_text[:60])

        yield "TXT", source_text, None, None
        transformed_text = self.text_transform_backend.transform(source_text, text_transform_prompt)
        logger.info("TXT: %.2fs %s", time.time() - t0, transformed_text[:60])

        yield "VOX", source_text, transformed_text, None
        voice_context = self._voice_context_extractor.extract(audio_path)
        text_for_voice = transformed_text.strip() or source_text
        output_audio_path = self._step_voice(text_for_voice, audio_path, audio_key, voice_context)
        logger.info("VOX: %.2fs %s", time.time() - t0, output_audio_path)

        yield "DONE", source_text, transformed_text, output_audio_path

    def process_stream_live(
        self,
        audio_path: str,
        text_transform_prompt: TextTransformPrompt,
    ) -> Generator[tuple[str, str | None, str | None, str | None], None, None]:
        """Live streaming pipeline with partial STT results.
        
        Yields: (status, source_text, transformed_text, output_audio_path)
        Status: "STT_PARTIAL" | "STT" | "TXT" | "VOX" | "DONE"
        """
        t0 = time.time()
        self.check_audio(audio_path)
        cache_key = AudioCacheKey.from_file(audio_path)
        audio_key = AudioKeyFromCacheKey(cache_key)

        # Check cache first
        cached_stt = self.processing_cache.get_stt(audio_key)
        if cached_stt is not None:
            source_text = cached_stt
            yield "STT", source_text, None, None
            logger.info("STT (cached): %s", source_text[:60])
        else:
            # Stream STT
            if hasattr(self.stt_backend, "transcribe_stream"):
                accumulated = ""
                for partial in self.stt_backend.transcribe_stream(audio_path):
                    accumulated = partial
                    yield "STT_PARTIAL", partial, None, None
                source_text = accumulated
                self.processing_cache.set_stt(audio_key, source_text)
                yield "STT", source_text, None, None
                logger.info("STT: %.2fs %s", time.time() - t0, source_text[:60])
            else:
                source_text = self._step_stt(audio_path, audio_key)
                yield "STT", source_text, None, None
                logger.info("STT: %.2fs %s", time.time() - t0, source_text[:60])

        # Text transform
        yield "TXT", source_text, None, None
        transformed_text = self.text_transform_backend.transform(source_text, text_transform_prompt)
        logger.info("TXT: %.2fs %s", time.time() - t0, transformed_text[:60])

        # Voice output
        yield "VOX", source_text, transformed_text, None
        voice_context = self._voice_context_extractor.extract(audio_path)
        text_for_voice = transformed_text.strip() or source_text
        output_audio_path = self._step_voice(text_for_voice, audio_path, audio_key, voice_context)
        logger.info("VOX: %.2fs %s", time.time() - t0, output_audio_path)

        yield "DONE", source_text, transformed_text, output_audio_path
