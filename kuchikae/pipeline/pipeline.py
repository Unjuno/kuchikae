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
from kuchikae.domain.stt import DummySTTBackend, SegmentedSTTBackend, STTBackend
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
    VoiceOutputPrompt,
)
from kuchikae.domain.voice_output import (
    DummyVoiceOutputBackend,
    VoiceOutputBackend,
)
from kuchikae.logging import setup_logger
from kuchikae.pipeline.audio_validation import validate_audio

logger = setup_logger("kuchikae.pipeline")


def create_pipeline(backend_config: dict | None = None) -> KuchikaePipeline:
    config = backend_config or {}
    allow_dummy_backends = config.get("allow_dummy_backends", True)

    stt_type = config.get("stt_backend", "faster_whisper")
    use_segmented = config.get("segmented_stt", False)
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        has_faster_whisper = True
    except ImportError:
        has_faster_whisper = False

    if stt_type == "faster_whisper" and has_faster_whisper:
        from kuchikae.backends.stt import FasterWhisperSTTBackend
        inner = FasterWhisperSTTBackend()
    elif stt_type == "faster_whisper" and not allow_dummy_backends:
        raise RuntimeError(
            "faster-whisper is required for the selected STT backend, but it is not "
            "importable in the current uv environment. Run `uv sync --extra real` or "
            "set `allow_dummy_backends=true` for development-only fallback."
        )
    else:
        inner = DummySTTBackend()

    stt: STTBackend
    if use_segmented:
        segmenter: AudioSegmenter = FixedWindowSegmenter(chunk_sec=30.0, overlap_sec=2.0)
        stt = SegmentedSTTBackend(inner=inner, segmenter=segmenter)
    elif config.get("streaming_stt", False):
        from kuchikae.backends.stt import StreamingFasterWhisperSTTBackend
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
    text_strict = not allow_dummy_backends
    if tt_class == GPTTextTransformBackend:
        if not os.environ.get("OPENAI_API_KEY") and text_strict:
            raise RuntimeError(
                "GPT text transform was requested, but OPENAI_API_KEY is not set. "
                "Either set the key or use a non-GPT text backend."
            )
        tt_kwargs["strict"] = text_strict
    elif tt_class == OllamaTextTransformBackend:
        tt_kwargs["strict"] = text_strict

    voice_output_type = config.get("voice_output_backend", "irodori")
    _ow_path = os.environ.get("KUCHIKAE_OPENVOICE_PATH", "/Users/taka/repos/OpenVoice")
    _irodori_ready = False
    try:
        from irodori_tts.inference_runtime import InferenceRuntime  # noqa: F401
        _irodori_ready = True
    except ImportError:
        pass

    if voice_output_type == "irodori" and _irodori_ready:
        from kuchikae.backends.voice_output import IrodoriTTSVoiceOutputBackend
        vo = IrodoriTTSVoiceOutputBackend()
    elif voice_output_type == "irodori" and not allow_dummy_backends:
        raise RuntimeError(
            "Irodori-TTS backend was requested, but `irodori_tts` is not importable "
            "in the current uv environment. Run `uv sync --extra real` or install the "
            "voice backend dependencies."
        )
    elif voice_output_type == "openvoice" or (config.get("auto_openvoice") and os.environ.get("OPENVOICE_READY")):
        from kuchikae.backends.voice_output import OpenVoiceOutputBackend
        vo = OpenVoiceOutputBackend()
    elif voice_output_type == "openvoice" and not allow_dummy_backends:
        raise RuntimeError(
            "OpenVoice backend was requested, but the OpenVoice checkout is not "
            "available or OPENVOICE_READY is not set."
        )
    else:
        _ow_ready = os.path.isdir(_ow_path) and os.environ.get("OPENVOICE_READY")
        if _ow_ready:
            from kuchikae.backends.voice_output import OpenVoiceOutputBackend
            vo = OpenVoiceOutputBackend()
        elif not allow_dummy_backends:
            raise RuntimeError(
                "No real voice backend is available. Set OPENVOICE_READY or install "
                "irodori_tts / OpenVoice, or enable allow_dummy_backends for tests."
            )
        else:
            vo = DummyVoiceOutputBackend()

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

        if hasattr(self.stt_backend, "_load_model"):
            try:
                logger.info("warming up STT...")
                self.stt_backend._load_model()  # noqa: SLF001
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
        try:
            validate_audio(audio_path)
        except ValueError as e:
            if "not found" in str(e):
                raise FileNotFoundError(str(e)) from e
            raise

    def _step_stt(self, audio_path: str, audio_key: AudioKey) -> str:
        cached = self.processing_cache.get_stt(audio_key)
        if cached is not None:
            return cached
        text = self.stt_backend.transcribe(audio_path)
        self.processing_cache.set_stt(audio_key, text)
        return text

    def _step_voice(
        self,
        text: str,
        audio_path: str,
        audio_key: AudioKey,
        voice_context,
        voice_output_prompt: VoiceOutputPrompt | None = None,
    ) -> str:
        cached_context = self.processing_cache.get_voice_context(audio_key)
        if cached_context is not None:
            voice_context = cached_context
        else:
            self.processing_cache.set_voice_context(audio_key, voice_context)
        
        out = self.voice_output_backend.synthesize(text, voice_context, voice_output_prompt)
        return out

    def process(
        self,
        audio_path: str,
        text_transform_prompt: TextTransformPrompt,
        voice_output_prompt: VoiceOutputPrompt | None = None,
    ) -> PipelineResult:
        t0 = time.time()
        logger.info(
            "process:start audio_path=%s text_prompt_len=%d voice_prompt=%s stt=%s text=%s voice=%s",
            audio_path,
            len(text_transform_prompt.instruction),
            "set" if voice_output_prompt is not None else "none",
            type(self.stt_backend).__name__,
            type(self.text_transform_backend).__name__,
            type(self.voice_output_backend).__name__,
        )

        self.check_audio(audio_path)
        cache_key = AudioCacheKey.from_file(audio_path)
        audio_key = AudioKeyFromCacheKey(cache_key)
        logger.info(
            "process:audio_validated cache_key path=%s size=%d mtime=%.6f audio_key=%s",
            cache_key.path,
            cache_key.size,
            cache_key.mtime,
            audio_key,
        )

        t1 = time.time()
        logger.info("process:stt:start")
        source_text = self._step_stt(audio_path, audio_key)
        stt_latency = time.time() - t1
        logger.info(
            "process:stt:done latency=%.2fs source_len=%d source_preview=%r",
            stt_latency,
            len(source_text),
            source_text[:120],
        )

        t2 = time.time()
        logger.info(
            "process:text:start text_prompt_preview=%r",
            text_transform_prompt.instruction[:160],
        )
        transformed_text = self.text_transform_backend.transform(source_text, text_transform_prompt)
        text_latency = time.time() - t2
        logger.info(
            "process:text:done latency=%.2fs transformed_len=%d transformed_preview=%r",
            text_latency,
            len(transformed_text),
            transformed_text[:120],
        )

        t3 = time.time()
        logger.info("process:voice_context:start")
        voice_context = self._voice_context_extractor.extract(audio_path)
        logger.info(
            "process:voice_context:done reference=%r ready=%s has_embedding=%s has_prosody=%s",
            voice_context.reference_audio_path,
            voice_context.ready,
            voice_context.speaker_embedding is not None,
            voice_context.prosody_profile is not None,
        )
        text_for_voice = transformed_text.strip() or source_text
        logger.info(
            "process:voice:start text_for_voice_len=%d text_for_voice_preview=%r voice_prompt=%s",
            len(text_for_voice),
            text_for_voice[:120],
            "set" if voice_output_prompt is not None else "none",
        )
        output_audio_path = self._step_voice(
            text_for_voice,
            audio_path,
            audio_key,
            voice_context,
            voice_output_prompt,
        )
        voice_latency = time.time() - t3
        logger.info(
            "process:voice:done latency=%.2fs output_audio_path=%s",
            voice_latency,
            output_audio_path,
        )

        total = time.time() - t0
        logger.info(
            "process:done total=%.2fs stt=%.2fs text=%.2fs voice=%.2fs",
            total,
            stt_latency,
            text_latency,
            voice_latency,
        )

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
        voice_output_prompt: VoiceOutputPrompt | None = None,
    ) -> Generator[tuple[str, str | None, str | None, str | None], None, None]:
        t0 = time.time()
        logger.info(
            "process_stream:start audio_path=%s text_prompt_len=%d voice_prompt=%s stt=%s text=%s voice=%s",
            audio_path,
            len(text_transform_prompt.instruction),
            "set" if voice_output_prompt is not None else "none",
            type(self.stt_backend).__name__,
            type(self.text_transform_backend).__name__,
            type(self.voice_output_backend).__name__,
        )
        self.check_audio(audio_path)
        cache_key = AudioCacheKey.from_file(audio_path)
        audio_key = AudioKeyFromCacheKey(cache_key)
        logger.info(
            "process_stream:audio_validated cache_key path=%s size=%d mtime=%.6f audio_key=%s",
            cache_key.path,
            cache_key.size,
            cache_key.mtime,
            audio_key,
        )

        logger.info("process_stream:yield STT")
        yield "STT", None, None, None
        source_text = self._step_stt(audio_path, audio_key)
        logger.info(
            "process_stream:stt:done elapsed=%.2fs source_len=%d source_preview=%r",
            time.time() - t0,
            len(source_text),
            source_text[:120],
        )

        logger.info("process_stream:yield TXT")
        yield "TXT", source_text, None, None
        logger.info(
            "process_stream:text:start text_prompt_preview=%r",
            text_transform_prompt.instruction[:160],
        )
        transformed_text = self.text_transform_backend.transform(source_text, text_transform_prompt)
        logger.info(
            "process_stream:text:done elapsed=%.2fs transformed_len=%d transformed_preview=%r",
            time.time() - t0,
            len(transformed_text),
            transformed_text[:120],
        )

        logger.info("process_stream:yield VOX")
        yield "VOX", source_text, transformed_text, None
        logger.info("process_stream:voice_context:start")
        voice_context = self._voice_context_extractor.extract(audio_path)
        logger.info(
            "process_stream:voice_context:done reference=%r ready=%s has_embedding=%s has_prosody=%s",
            voice_context.reference_audio_path,
            voice_context.ready,
            voice_context.speaker_embedding is not None,
            voice_context.prosody_profile is not None,
        )
        text_for_voice = transformed_text.strip() or source_text
        logger.info(
            "process_stream:voice:start text_for_voice_len=%d text_for_voice_preview=%r voice_prompt=%s",
            len(text_for_voice),
            text_for_voice[:120],
            "set" if voice_output_prompt is not None else "none",
        )
        output_audio_path = self._step_voice(
            text_for_voice,
            audio_path,
            audio_key,
            voice_context,
            voice_output_prompt,
        )
        logger.info(
            "process_stream:voice:done elapsed=%.2fs output_audio_path=%s",
            time.time() - t0,
            output_audio_path,
        )

        logger.info("process_stream:yield DONE")
        yield "DONE", source_text, transformed_text, output_audio_path

    def process_stream_live(
        self,
        audio_path: str,
        text_transform_prompt: TextTransformPrompt,
        voice_output_prompt: VoiceOutputPrompt | None = None,
    ) -> Generator[tuple[str, str | None, str | None, str | None], None, None]:
        """Live streaming pipeline with partial STT results.
        
        Yields: (status, source_text, transformed_text, output_audio_path)
        Status: "STT_PARTIAL" | "STT" | "TXT" | "VOX" | "DONE"
        """
        t0 = time.time()
        logger.info(
            "process_stream_live:start audio_path=%s text_prompt_len=%d voice_prompt=%s stt=%s text=%s voice=%s",
            audio_path,
            len(text_transform_prompt.instruction),
            "set" if voice_output_prompt is not None else "none",
            type(self.stt_backend).__name__,
            type(self.text_transform_backend).__name__,
            type(self.voice_output_backend).__name__,
        )
        self.check_audio(audio_path)
        cache_key = AudioCacheKey.from_file(audio_path)
        audio_key = AudioKeyFromCacheKey(cache_key)
        logger.info(
            "process_stream_live:audio_validated cache_key path=%s size=%d mtime=%.6f audio_key=%s",
            cache_key.path,
            cache_key.size,
            cache_key.mtime,
            audio_key,
        )

        # Check cache first
        cached_stt = self.processing_cache.get_stt(audio_key)
        if cached_stt is not None:
            logger.info(
                "process_stream_live:stt_cache_hit source_len=%d source_preview=%r",
                len(cached_stt),
                cached_stt[:120],
            )
            source_text = cached_stt
            logger.info("process_stream_live:yield STT")
            yield "STT", source_text, None, None
        else:
            # Stream STT
            if hasattr(self.stt_backend, "transcribe_stream"):
                logger.info("process_stream_live:stt_stream:start backend=%s", type(self.stt_backend).__name__)
                accumulated = ""
                for idx, partial in enumerate(self.stt_backend.transcribe_stream(audio_path), start=1):
                    accumulated = partial
                    logger.info(
                        "process_stream_live:stt_partial idx=%d partial_len=%d partial_preview=%r",
                        idx,
                        len(partial),
                        partial[:120],
                    )
                    yield "STT_PARTIAL", partial, None, None
                source_text = accumulated
                self.processing_cache.set_stt(audio_key, source_text)
                logger.info(
                    "process_stream_live:stt_stream:done source_len=%d source_preview=%r",
                    len(source_text),
                    source_text[:120],
                )
                logger.info("process_stream_live:yield STT")
                yield "STT", source_text, None, None
            else:
                logger.info("process_stream_live:stt_fallback:start backend=%s", type(self.stt_backend).__name__)
                source_text = self._step_stt(audio_path, audio_key)
                logger.info(
                    "process_stream_live:stt_fallback:done source_len=%d source_preview=%r",
                    len(source_text),
                    source_text[:120],
                )
                logger.info("process_stream_live:yield STT")
                yield "STT", source_text, None, None

        # Text transform
        logger.info("process_stream_live:yield TXT")
        yield "TXT", source_text, None, None
        logger.info(
            "process_stream_live:text:start text_prompt_preview=%r",
            text_transform_prompt.instruction[:160],
        )
        transformed_text = self.text_transform_backend.transform(source_text, text_transform_prompt)
        logger.info(
            "process_stream_live:text:done elapsed=%.2fs transformed_len=%d transformed_preview=%r",
            time.time() - t0,
            len(transformed_text),
            transformed_text[:120],
        )

        # Voice output
        logger.info("process_stream_live:yield VOX")
        yield "VOX", source_text, transformed_text, None
        logger.info("process_stream_live:voice_context:start")
        voice_context = self._voice_context_extractor.extract(audio_path)
        logger.info(
            "process_stream_live:voice_context:done reference=%r ready=%s has_embedding=%s has_prosody=%s",
            voice_context.reference_audio_path,
            voice_context.ready,
            voice_context.speaker_embedding is not None,
            voice_context.prosody_profile is not None,
        )
        text_for_voice = transformed_text.strip() or source_text
        logger.info(
            "process_stream_live:voice:start text_for_voice_len=%d text_for_voice_preview=%r voice_prompt=%s",
            len(text_for_voice),
            text_for_voice[:120],
            "set" if voice_output_prompt is not None else "none",
        )
        output_audio_path = self._step_voice(
            text_for_voice,
            audio_path,
            audio_key,
            voice_context,
            voice_output_prompt,
        )
        logger.info(
            "process_stream_live:voice:done elapsed=%.2fs output_audio_path=%s",
            time.time() - t0,
            output_audio_path,
        )

        logger.info("process_stream_live:yield DONE")
        yield "DONE", source_text, transformed_text, output_audio_path
