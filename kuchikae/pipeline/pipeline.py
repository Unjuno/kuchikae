"""Kuchikae pipeline orchestration."""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Generator

from kuchikae.domain.audio import AudioSegmenter, FixedWindowSegmenter
from kuchikae.domain.audio_cache import AudioCache, VoiceContextExtractor, DummyVoiceContextExtractor
from kuchikae.domain.audio_key import AudioKey, AudioKeyFromCacheKey
from kuchikae.domain.metrics import LatencyLogger
from kuchikae.domain.diagnostics import DiagnosticRecorder
from kuchikae.domain.error_hints import hint_for_error
from kuchikae.domain.events import DiagnosticEvent, EventLevel, new_run_id
from kuchikae.counting_backends import (
    CountingSTTBackend,
    CountingTextTransformBackend,
    CountingVoiceContextExtractor,
    CountingVoiceOutputBackend,
)
from kuchikae.domain.processing_cache import ProcessingCache
from kuchikae.domain.audio_emotion import (
    AudioEmotion,
    AudioEmotionDetector,
    DisabledAudioEmotionDetector,
    DummyAudioEmotionDetector,
    TransformersAudioEmotionDetector,
)
from kuchikae.domain.stt import (
    DummySTTBackend,
    FasterWhisperConfig,
    SegmentedSTTBackend,
    STTBackend,
    resolve_stt_preset,
)
from kuchikae.domain.text_transform import (
    DummyTextTransformBackend,
    GPTTextTransformBackend,
    OllamaTextTransformBackend,
    PromptedRuleTextTransformBackend,
    RuleTextTransformBackend,
    TextTransformBackend,
    strip_cot,
    validate_transform,
)
from kuchikae.domain.voice_style import (
    RuleVoiceStyleDetector,
    VoiceStyle,
    VoiceStyleDetector,
    merge_voice_style,
    voice_style_to_prompt,
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
    allow_dummy_backends = config.get("allow_dummy_backends", False)
    stt_preset_name = config.get("stt_preset", "balanced")
    stt_preset = resolve_stt_preset(stt_preset_name)
    disable_processing_cache = bool(
        config.get("disable_processing_cache", False)
        or os.environ.get("KUCHIKAE_DISABLE_PROCESSING_CACHE", "").lower() in ("1", "true", "yes")
    )

    stt_type = config.get("stt_backend", "faster_whisper")
    use_segmented = config.get("segmented_stt", False)
    has_faster_whisper = False
    try:
        from faster_whisper import WhisperModel  # noqa: F401

        has_faster_whisper = True
    except ImportError:
        pass
    has_transformers = False
    try:
        from transformers import AutoProcessor  # noqa: F401

        has_transformers = True
    except ImportError:
        pass
    has_nemo = False
    try:
        import nemo.collections.asr  # noqa: F401

        has_nemo = True
    except ImportError:
        pass

    if stt_type == "faster_whisper" and has_faster_whisper:
        from kuchikae.backends.stt import FasterWhisperSTTBackend
        stt_config = FasterWhisperConfig(
            model_size=config.get("stt_model_size", stt_preset.model_size),
            device=config.get("stt_device", stt_preset.device),
            compute_type=config.get("stt_compute_type", stt_preset.compute_type),
            language=config.get("stt_language", stt_preset.language),
            beam_size=int(config.get("stt_beam_size", stt_preset.beam_size)),
            vad_filter=bool(config.get("stt_vad_filter", stt_preset.vad_filter)),
            temperature=float(config.get("stt_temperature", stt_preset.temperature)),
            condition_on_previous_text=bool(
                config.get("stt_condition_on_previous_text", stt_preset.condition_on_previous_text)
            ),
        )
        inner = FasterWhisperSTTBackend(config=stt_config)
    elif stt_type == "faster_whisper" and not allow_dummy_backends:
        raise RuntimeError(
            "faster-whisper is required for the selected STT backend, but it is not "
            "importable in the current uv environment. Run `uv sync --extra real` or "
            "set `allow_dummy_backends=true` for development-only fallback."
        )
    elif stt_type == "transformers_japanese" and has_transformers:
        from kuchikae.backends.stt_transformers import TransformersJapaneseASRBackend
        inner = TransformersJapaneseASRBackend(
            model_id=config.get("stt_model_id"),
            device=config.get("stt_device"),
            torch_dtype=config.get("stt_torch_dtype"),
        )
    elif stt_type == "transformers_japanese" and not allow_dummy_backends:
        raise RuntimeError(
            "Transformers Japanese STT backend was requested, but `transformers` is not "
            "importable in the current uv environment. Run `uv sync --extra real` or "
            "set `allow_dummy_backends=true` for development-only fallback."
        )
    elif stt_type == "transformers_whisper" and has_transformers:
        from kuchikae.backends.stt_transformers_whisper import (
            TransformersWhisperJapaneseASRBackend,
        )
        inner = TransformersWhisperJapaneseASRBackend(
            model_id=config.get("stt_model_id"),
            device=config.get("stt_device"),
            torch_dtype=config.get("stt_torch_dtype"),
        )
    elif stt_type == "transformers_whisper" and not allow_dummy_backends:
        raise RuntimeError(
            "Transformers Whisper STT backend was requested, but `transformers` is not "
            "importable in the current uv environment. Run `uv sync --extra real` or "
            "set `allow_dummy_backends=true` for development-only fallback."
        )
    elif stt_type == "reazonspeech_nemo" and has_nemo:
        from kuchikae.backends.stt_nemo import ReazonSpeechNemoASRBackend
        inner = ReazonSpeechNemoASRBackend()
    elif stt_type == "reazonspeech_nemo" and not allow_dummy_backends:
        raise RuntimeError(
            "reazonspeech_nemo was requested, but the required NeMo packages are not "
            "importable in the current uv environment."
        )
    elif allow_dummy_backends:
        inner = DummySTTBackend()
    else:
        raise RuntimeError(
            f"Unknown or unavailable STT backend: {stt_type}. "
            "Set allow_dummy_backends=true only for development fallback."
        )

    stt: STTBackend
    stt_config: FasterWhisperConfig | None = None
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
    _ow_path = os.environ.get("KUCHIKAE_OPENVOICE_PATH", "")
    _irodori_ready = False
    try:
        from irodori_tts.inference_runtime import InferenceRuntime  # noqa: F401
        _irodori_ready = True
    except ImportError:
        pass

    audio_emotion_detector_type = config.get("audio_emotion_detector", "dummy")
    if audio_emotion_detector_type == "transformers_audio_emotion":
        audio_emotion_detector = TransformersAudioEmotionDetector(
            model_id=config.get("audio_emotion_model_id"),
            strict=bool(config.get("audio_emotion_strict", False)),
        )
    elif audio_emotion_detector_type == "disabled":
        audio_emotion_detector = DisabledAudioEmotionDetector()
    else:
        audio_emotion_detector = DummyAudioEmotionDetector()

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
        _ow_ready = _ow_path and os.path.isdir(_ow_path) and os.environ.get("OPENVOICE_READY")
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

    logger.info(
        "pipeline: stt=%s preset=%s config=%s text=%s(%s) voice=%s cache=%s",
        type(stt).__name__,
        stt_preset_name,
        stt_config if stt_type == "faster_whisper" and has_faster_whisper else None,
        type(tt_class()).__name__,
        text_model or "default",
        type(vo).__name__,
        "disabled" if disable_processing_cache else "enabled",
    )

    if isinstance(stt, DummySTTBackend):
        logger.warning("backend.dummy_selected stage=stt message='Dummy STT selected; real audio will not be transcribed'")

    return KuchikaePipeline(
        stt_backend=stt,
        text_transform_backend=tt_class(**tt_kwargs),
        voice_output_backend=vo,
        disable_processing_cache=disable_processing_cache,
        stt_preset=stt_preset_name,
        stt_config=stt_config if stt_type == "faster_whisper" and has_faster_whisper else None,
        audio_emotion_detector=audio_emotion_detector,
        backend_config=config,
    )


class KuchikaePipeline:

    def __init__(
        self,
        stt_backend: STTBackend | None = None,
        text_transform_backend: TextTransformBackend | None = None,
        voice_output_backend: VoiceOutputBackend | None = None,
        processing_cache: ProcessingCache | None = None,
        latency_logger: LatencyLogger | None = None,
        disable_processing_cache: bool = False,
        diagnostics: DiagnosticRecorder | None = None,
        stt_preset: str = "balanced",
        stt_config: FasterWhisperConfig | None = None,
        voice_style_detector: VoiceStyleDetector | None = None,
        audio_emotion_detector: AudioEmotionDetector | None = None,
        voice_style_timeout_sec: float = 0.05,
        backend_config: dict | None = None,
    ) -> None:
        self.stt_backend = stt_backend or DummySTTBackend()
        self.text_transform_backend = text_transform_backend or DummyTextTransformBackend()
        if isinstance(self.text_transform_backend, OllamaTextTransformBackend):
            self.text_transform_backend._on_cot_stripped = self._on_cot_stripped
        self.voice_output_backend = voice_output_backend or DummyVoiceOutputBackend()
        self.processing_cache = processing_cache or ProcessingCache()
        self.latency_logger = latency_logger
        self.disable_processing_cache = disable_processing_cache
        self.diagnostics = diagnostics or DiagnosticRecorder()
        self.run_id = new_run_id()
        self.stt_preset = stt_preset
        self.stt_config = stt_config
        self.voice_style_detector = voice_style_detector or RuleVoiceStyleDetector()
        self.audio_emotion_detector = audio_emotion_detector or DummyAudioEmotionDetector()
        self.voice_style_timeout_sec = voice_style_timeout_sec
        self._last_voice_style = "auto"
        self._last_audio_emotion = "disabled"
        self.backend_config = backend_config or {}
        self._audio_cache = AudioCache()
        self._voice_context_extractor = VoiceContextExtractor()
        self.diagnostics.emit(
            DiagnosticEvent(
                name="backend.selected",
                stage="pipeline",
                backend=type(self.stt_backend).__name__,
                message=(
                    f"STT={type(self.stt_backend).__name__}, "
                    f"TEXT={type(self.text_transform_backend).__name__}, "
                    f"VOICE={type(self.voice_output_backend).__name__}"
                ),
                run_id=self.run_id,
                data={
                    "stt": type(self.stt_backend).__name__,
                    "text": type(self.text_transform_backend).__name__,
                    "voice": type(self.voice_output_backend).__name__,
                    "cache": "disabled" if self.disable_processing_cache else "enabled",
                    "stt_preset": self.stt_preset,
                    "stt_config": self._stt_config_data(),
                },
            )
        )
        if isinstance(self.stt_backend, DummySTTBackend):
            self.diagnostics.emit(
                DiagnosticEvent(
                    name="backend.dummy_selected",
                    level=EventLevel.WARNING,
                    stage="stt",
                    backend=type(self.stt_backend).__name__,
                    message="STT backend is DummySTTBackend. 実音声は認識されません。",
                    run_id=self.run_id,
                    data={"allow_dummy_backends": True},
                )
            )

    def _cache_enabled(self) -> bool:
        return not self.disable_processing_cache

    def _emit(self, name: str, message: str, stage: str, level: EventLevel = EventLevel.INFO, backend: str | None = None, cache: str | None = None, elapsed_sec: float | None = None, data: dict | None = None) -> None:
        self.diagnostics.emit(
            DiagnosticEvent(
                name=name,
                level=level,
                message=message,
                run_id=self.run_id,
                stage=stage,
                backend=backend,
                cache=cache,
                elapsed_sec=elapsed_sec,
                data=data or {},
            )
        )

    def _emit_failed(self, stage: str, backend: str, error: Exception) -> None:
        self._emit(
            "pipeline.failed",
            f"{type(error).__name__}: {error}",
            "pipeline",
            level=EventLevel.ERROR,
            backend=backend,
            data={"stage": stage, "hint": hint_for_error(stage, error)},
        )

    def _emit_style(self, name: str, stage: str, data: dict) -> None:
        self._emit(name, name, stage, data=data)

    def _on_cot_stripped(self, model: str) -> None:
        self._emit(
            "text_transform.cot_stripped",
            "CoT content was stripped from transform result.",
            "text",
            level=EventLevel.WARNING,
            backend="OllamaTextTransformBackend",
            data={"model": model},
        )

    def _stt_config_data(self) -> dict:
        config = getattr(self.stt_backend, "config", None)
        if config is not None:
            return {
                "model_size": getattr(config, "model_size", None),
                "device": getattr(config, "device", None),
                "compute_type": getattr(config, "compute_type", None),
                "language": getattr(config, "language", None),
                "beam_size": getattr(config, "beam_size", None),
                "vad_filter": getattr(config, "vad_filter", None),
                "temperature": getattr(config, "temperature", None),
                "condition_on_previous_text": getattr(config, "condition_on_previous_text", None),
            }
        return {
            "model_size": getattr(self.stt_backend, "model_size", None),
            "device": getattr(self.stt_backend, "device", None),
            "compute_type": getattr(self.stt_backend, "compute_type", None),
        }

    def set_stt_preset(self, stt_preset: str) -> None:
        from kuchikae.backends.stt import FasterWhisperSTTBackend

        config = resolve_stt_preset(stt_preset)
        old_preset = self.stt_preset
        old_config = self.stt_config
        if isinstance(self.stt_backend, FasterWhisperSTTBackend):
            self.stt_backend = FasterWhisperSTTBackend(config=config)
        elif hasattr(self.stt_backend, "_inner") and isinstance(getattr(self.stt_backend, "_inner"), FasterWhisperSTTBackend):
            self.stt_backend._inner = FasterWhisperSTTBackend(config=config)  # type: ignore[attr-defined]
        self.stt_preset = stt_preset
        self.stt_config = config
        self._emit(
            "backend.selected",
            f"STT preset changed to {stt_preset}",
            "pipeline",
            data={
                "stt": type(self.stt_backend).__name__,
                "stt_preset": self.stt_preset,
                "stt_config": self._stt_config_data(),
                "previous_stt_preset": old_preset,
                "previous_stt_config": asdict(old_config) if old_config is not None else None,
                "cache_reuse_warning": "cached STT/text/voice outputs may reflect the previous preset",
            },
        )
        if isinstance(self.stt_backend, DummySTTBackend):
            self._emit(
                "backend.dummy_selected",
                "Dummy STT selected; real audio will not be transcribed.",
                "stt",
                level=EventLevel.WARNING,
                data={"allow_dummy_backends": True, "stt_preset": self.stt_preset},
            )

    def warmup(self) -> None:
        logger.info("warming up...")

        if os.environ.get("KUCHIKAE_SKIP_WARMUP", "").lower() in ("1", "true", "yes"):
            logger.info("warmup skipped by KUCHIKAE_SKIP_WARMUP")
            return

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
                    timeout=float(os.environ.get("KUCHIKAE_WARMUP_TIMEOUT", "2.5")),
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
        cached = None if self.disable_processing_cache else self.processing_cache.get_stt(audio_key)
        if cached is not None:
            self._emit(
                "cache.stt_hit",
                "STT cache hit; backend was not executed.",
                "stt",
                cache="stt",
                data={"source_len": len(cached), "source_preview": cached[:80]},
            )
            return cached
        text = self.stt_backend.transcribe(audio_path)
        if not self.disable_processing_cache:
            self.processing_cache.set_stt(audio_key, text)
        return text

    def _step_stt_stream(self, audio_path: str, audio_key: AudioKey) -> str:
        self._emit(
            "stt.start",
            "STT started.",
            "stt",
            backend=type(self.stt_backend).__name__,
        )
        if hasattr(self.stt_backend, "transcribe_stream"):
            accumulated = ""
            for partial in self.stt_backend.transcribe_stream(audio_path):
                accumulated = partial
            if not self.disable_processing_cache:
                self.processing_cache.set_stt(audio_key, accumulated)
            return accumulated
        return self._step_stt(audio_path, audio_key)

    def _voice_context(self, audio_path: str, audio_key: AudioKey) -> object:
        if self.disable_processing_cache:
            return self._voice_context_extractor.extract(audio_path)
        cached = self.processing_cache.get_voice_context(audio_key)
        if cached is not None:
            self._emit(
                "cache.voice_hit",
                "Voice context cache hit; extractor was not executed.",
                "voice_context",
                cache="voice_context",
                data={"reference_audio_path": cached.reference_audio_path, "ready": cached.ready},
            )
            return cached
        voice_context = self._voice_context_extractor.extract(audio_path)
        self.processing_cache.set_voice_context(audio_key, voice_context)
        return voice_context

    def _detect_audio_emotion_async(self, audio_path: str):
        if self.audio_emotion_detector is None or getattr(self.audio_emotion_detector, "disabled", False):
            return None, None
        executor = ThreadPoolExecutor(max_workers=1)
        self._emit(
            "audio_emotion.detect.start",
            "Audio emotion detection started.",
            "audio_emotion",
            backend=type(self.audio_emotion_detector).__name__,
        )
        future = executor.submit(self.audio_emotion_detector.detect, audio_path)
        return future, executor

    def _collect_audio_emotion(self, audio_future, audio_executor):
        audio_emotion = None
        if audio_future is None:
            return None
        try:
            audio_emotion = audio_future.result(timeout=self.voice_style_timeout_sec)
            self._emit(
                "audio_emotion.detect.done",
                "Audio emotion detection finished.",
                "audio_emotion",
                backend=type(self.audio_emotion_detector).__name__ if self.audio_emotion_detector is not None else "disabled",
                data=asdict(audio_emotion),
            )
        except FuturesTimeoutError:
            self._emit(
                "audio_emotion.detect.timeout",
                "Audio emotion detection timed out.",
                "audio_emotion",
                level=EventLevel.WARNING,
                backend=type(self.audio_emotion_detector).__name__ if self.audio_emotion_detector is not None else "disabled",
            )
        except Exception as e:
            self._emit(
                "audio_emotion.detect.failed",
                f"{type(e).__name__}: {e}",
                "audio_emotion",
                level=EventLevel.WARNING,
                backend=type(self.audio_emotion_detector).__name__ if self.audio_emotion_detector is not None else "disabled",
            )
        finally:
            if audio_executor is not None:
                audio_executor.shutdown(wait=False, cancel_futures=True)
        return audio_emotion

    def _detect_voice_style(self, text: str) -> VoiceStyle:
        self._emit_style("voice_style.detect.start", "voice_style", {})
        style = self.voice_style_detector.detect(text)
        self._emit_style(
            "voice_style.detect.done",
            "voice_style",
            {
                "mood": style.mood.value,
                "speed": style.speed.value,
                "clarity": style.clarity.value,
                "emphasis": style.emphasis.value,
                "confidence": style.confidence,
                "source": style.source,
            },
        )
        return style

    def _build_voice_prompt(self, text_for_voice: str, audio_emotion: AudioEmotion | None, voice_output_prompt: VoiceOutputPrompt | None) -> VoiceOutputPrompt:
        if voice_output_prompt is not None:
            self._last_voice_style = "custom"
            self._last_audio_emotion = getattr(audio_emotion, "source", "disabled") if audio_emotion is not None else "disabled"
            return voice_output_prompt
        text_style = self._detect_voice_style(text_for_voice)
        final_style = merge_voice_style(text_style, audio_emotion)
        prompt = VoiceOutputPrompt(instruction=voice_style_to_prompt(final_style))
        self._last_voice_style = final_style.mood.value
        self._last_audio_emotion = getattr(audio_emotion, "source", "disabled") if audio_emotion is not None else "disabled"
        self._emit_style(
            "voice_style.generated",
            "voice_style",
            {
                "mood": final_style.mood.value,
                "speed": final_style.speed.value,
                "clarity": final_style.clarity.value,
                "emphasis": final_style.emphasis.value,
                "confidence": final_style.confidence,
                "source": final_style.source,
                "generated_prompt_preview": prompt.instruction[:120],
            },
        )
        return prompt

    def _step_voice(
        self,
        text: str,
        audio_path: str,
        audio_key: AudioKey,
        voice_context,
        voice_output_prompt: VoiceOutputPrompt | None = None,
    ) -> str:
        if self.disable_processing_cache:
            return self.voice_output_backend.synthesize(text, voice_context, voice_output_prompt)
        cached_context = self.processing_cache.get_voice_context(audio_key)
        if cached_context is not None:
            voice_context = cached_context
        else:
            self.processing_cache.set_voice_context(audio_key, voice_context)
        return self.voice_output_backend.synthesize(text, voice_context, voice_output_prompt)

    def process(
        self,
        audio_path: str,
        text_transform_prompt: TextTransformPrompt,
        voice_output_prompt: VoiceOutputPrompt | None = None,
    ) -> PipelineResult:
        t0 = time.time()
        logger.info(
            "process:start audio_path=%s text_prompt_len=%d voice_prompt=%s stt=%s text=%s voice=%s cache=%s",
            audio_path,
            len(text_transform_prompt.instruction),
            "set" if voice_output_prompt is not None else "none",
            type(self.stt_backend).__name__,
            type(self.text_transform_backend).__name__,
            type(self.voice_output_backend).__name__,
            "disabled" if self.disable_processing_cache else "enabled",
        )

        self.check_audio(audio_path)
        cache_key = AudioCacheKey.from_file(audio_path)
        audio_key = AudioKeyFromCacheKey(cache_key)
        voice_prompt_text = voice_output_prompt.instruction if voice_output_prompt is not None else ""
        cached_result = None
        if not self.disable_processing_cache and voice_output_prompt is not None:
            cached_result = self.processing_cache.get_result(audio_key, text_transform_prompt.instruction, voice_prompt_text)
        if cached_result is not None:
            logger.info(
                "process:full_cache_hit output_audio_path=%s source_len=%d transformed_len=%d",
                cached_result.output_audio_path,
                len(cached_result.source_text),
                len(cached_result.transformed_text),
            )
            self._emit(
                "cache.full_hit",
                "Full pipeline cache hit; STT/text/TTS were not executed.",
                "pipeline",
                cache="full_result",
                data={
                    "source_len": len(cached_result.source_text),
                    "transformed_len": len(cached_result.transformed_text),
                    "output_audio_path": cached_result.output_audio_path,
                },
            )
            return cached_result
        logger.info(
            "process:audio_validated cache_key path=%s size=%d mtime=%.6f audio_key=%s",
            cache_key.path,
            cache_key.size,
            cache_key.mtime,
            audio_key,
        )
        audio_future, audio_executor = self._detect_audio_emotion_async(audio_path)

        t1 = time.time()
        self._emit(
            "stt.start",
            "STT started.",
            "stt",
            backend=type(self.stt_backend).__name__,
        )
        logger.info("process:stt:start")
        try:
            source_text = self._step_stt(audio_path, audio_key)
            stt_latency = time.time() - t1
            self._emit(
                "stt.done",
                "STT finished.",
                "stt",
                backend=type(self.stt_backend).__name__,
                elapsed_sec=stt_latency,
                data={"source_len": len(source_text), "source_preview": source_text[:80]},
            )
        except Exception as e:
            self._emit_failed("stt", type(self.stt_backend).__name__, e)
            self._emit(
                "stt.failed",
                f"{type(e).__name__}: {e}",
                "stt",
                level=EventLevel.ERROR,
                backend=type(self.stt_backend).__name__,
                data={"hint": hint_for_error("stt", e)},
            )
            raise
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
        cached_text = None if self.disable_processing_cache else self.processing_cache.get_text(source_text, text_transform_prompt)
        if cached_text is not None:
            transformed_text = cached_text
            logger.info("process:text:cache_hit transformed_len=%d", len(transformed_text))
            self._emit(
                "cache.text_hit",
                "Text transform cache hit; backend was not executed.",
                "text",
                cache="text",
                data={"transformed_len": len(transformed_text)},
            )
        else:
            try:
                self._emit(
                    "text.start",
                    "Text transform started.",
                    "text",
                    backend=type(self.text_transform_backend).__name__,
                )
                transformed_text = self.text_transform_backend.transform(source_text, text_transform_prompt)
                if not validate_transform(source_text, transformed_text):
                    self._emit(
                        "text_transform.validation_failed",
                        "Text transform validation failed.",
                        "text",
                        level=EventLevel.WARNING,
                        data={"source_len": len(source_text), "transformed_len": len(transformed_text)},
                    )
                    transformed_text = PromptedRuleTextTransformBackend().transform(source_text, text_transform_prompt)
                    self._emit(
                        "text_transform.fallback_used",
                        "PromptedRuleTextTransformBackend fallback used.",
                        "text",
                    )
                self._emit(
                    "text.done",
                    "Text transform finished.",
                    "text",
                    backend=type(self.text_transform_backend).__name__,
                    data={"transformed_len": len(transformed_text), "transformed_preview": transformed_text[:80]},
                )
            except Exception as e:
                self._emit_failed("text", type(self.text_transform_backend).__name__, e)
                self._emit(
                    "text.failed",
                    f"{type(e).__name__}: {e}",
                    "text",
                    level=EventLevel.ERROR,
                    backend=type(self.text_transform_backend).__name__,
                    data={"hint": hint_for_error("text", e)},
                )
                raise
            if not self.disable_processing_cache:
                self.processing_cache.set_text(source_text, text_transform_prompt, transformed_text)
        text_latency = time.time() - t2
        logger.info(
            "process:text:done latency=%.2fs transformed_len=%d transformed_preview=%r",
            text_latency,
            len(transformed_text),
            transformed_text[:120],
        )

        t3 = time.time()
        logger.info("process:voice_context:start")
        try:
            self._emit(
                "voice_context.start",
                "Voice context extraction started.",
                "voice_context",
                backend=type(self._voice_context_extractor).__name__,
            )
            voice_context = self._voice_context_extractor.extract(audio_path)
            self._emit(
                "voice_context.done",
                "Voice context extraction finished.",
                "voice_context",
                backend=type(self._voice_context_extractor).__name__,
                data={
                    "reference_audio_path": voice_context.reference_audio_path,
                    "ready": voice_context.ready,
                },
            )
        except Exception as e:
            self._emit_failed("voice_context", type(self._voice_context_extractor).__name__, e)
            self._emit(
                "voice_context.failed",
                f"{type(e).__name__}: {e}",
                "voice_context",
                level=EventLevel.ERROR,
                backend=type(self._voice_context_extractor).__name__,
                data={"hint": hint_for_error("voice_context", e)},
            )
            raise
        logger.info(
            "process:voice_context:done reference=%r ready=%s has_embedding=%s has_prosody=%s",
            voice_context.reference_audio_path,
            voice_context.ready,
            voice_context.speaker_embedding is not None,
            voice_context.prosody_profile is not None,
        )
        text_for_voice = transformed_text.strip() or source_text
        audio_emotion = self._collect_audio_emotion(audio_future, audio_executor)
        voice_output_prompt = self._build_voice_prompt(text_for_voice, audio_emotion, voice_output_prompt)
        voice_prompt_text = voice_output_prompt.instruction
        logger.info(
            "process:voice:start text_for_voice_len=%d text_for_voice_preview=%r voice_prompt=%s",
            len(text_for_voice),
            text_for_voice[:120],
            "set" if voice_output_prompt is not None else "none",
        )
        cached_audio = None if self.disable_processing_cache else self.processing_cache.get_voice_output(text_for_voice, voice_prompt_text, voice_context)
        if cached_audio is not None and os.path.exists(cached_audio):
            output_audio_path = cached_audio
            logger.info("process:voice:cache_hit output_audio_path=%s", output_audio_path)
            self._emit(
                "cache.voice_hit",
                "Voice output cache hit; backend was not executed.",
                "tts",
                cache="voice_output",
                data={"output_audio_path": output_audio_path},
            )
        else:
            try:
                self._emit(
                    "tts.start",
                    "Voice output started.",
                    "tts",
                    backend=type(self.voice_output_backend).__name__,
                )
                output_audio_path = self._step_voice(
                    text_for_voice,
                    audio_path,
                    audio_key,
                    voice_context,
                    voice_output_prompt,
                )
                self._emit(
                    "tts.done",
                    "Voice output finished.",
                    "tts",
                    backend=type(self.voice_output_backend).__name__,
                    data={"output_audio_path": output_audio_path},
                )
            except Exception as e:
                self._emit_failed("tts", type(self.voice_output_backend).__name__, e)
                self._emit(
                    "tts.failed",
                    f"{type(e).__name__}: {e}",
                    "tts",
                    level=EventLevel.ERROR,
                    backend=type(self.voice_output_backend).__name__,
                    data={"hint": hint_for_error("tts", e)},
                )
                raise
            if not self.disable_processing_cache:
                self.processing_cache.set_voice_output(
                    text_for_voice,
                    voice_prompt_text,
                    voice_context,
                    output_audio_path,
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

        if not self.disable_processing_cache:
            self.processing_cache.set_result(audio_key, text_transform_prompt.instruction, voice_prompt_text, result)
        self._emit(
            "pipeline.done",
            "Pipeline finished.",
            "pipeline",
            data={"total_latency": total, "source_len": len(source_text), "transformed_len": len(transformed_text)},
        )
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
        voice_prompt_text = voice_output_prompt.instruction if voice_output_prompt is not None else ""
        cached_result = None
        if not self.disable_processing_cache and voice_output_prompt is not None:
            cached_result = self.processing_cache.get_result(audio_key, text_transform_prompt.instruction, voice_prompt_text)
        if cached_result is not None:
            logger.info("process_stream:full_cache_hit")
            self._emit(
                "cache.full_hit",
                "Full pipeline cache hit; STT/text/TTS were not executed.",
                "pipeline",
                cache="full_result",
                data={
                    "source_len": len(cached_result.source_text),
                    "transformed_len": len(cached_result.transformed_text),
                    "output_audio_path": cached_result.output_audio_path,
                },
            )
            yield "STT", cached_result.source_text, None, None
            yield "TXT", cached_result.source_text, cached_result.transformed_text, None
            yield "VOX", cached_result.source_text, cached_result.transformed_text, None
            yield "DONE", cached_result.source_text, cached_result.transformed_text, cached_result.output_audio_path
            return
        logger.info(
            "process_stream:audio_validated cache_key path=%s size=%d mtime=%.6f audio_key=%s",
            cache_key.path,
            cache_key.size,
            cache_key.mtime,
            audio_key,
        )
        audio_future, audio_executor = self._detect_audio_emotion_async(audio_path)

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
        cached_text = None if self.disable_processing_cache else self.processing_cache.get_text(source_text, text_transform_prompt)
        if cached_text is not None:
            transformed_text = cached_text
            logger.info("process_stream:text:cache_hit transformed_len=%d", len(transformed_text))
        else:
            transformed_text = self.text_transform_backend.transform(source_text, text_transform_prompt)
            if not validate_transform(source_text, transformed_text):
                self._emit("text_transform.validation_failed", "Text transform validation failed.", "text", level=EventLevel.WARNING, data={"source_len": len(source_text), "transformed_len": len(transformed_text)})
                transformed_text = PromptedRuleTextTransformBackend().transform(source_text, text_transform_prompt)
                self._emit("text_transform.fallback_used", "PromptedRuleTextTransformBackend fallback used.", "text")
            if not self.disable_processing_cache:
                self.processing_cache.set_text(source_text, text_transform_prompt, transformed_text)
        logger.info(
            "process_stream:text:done elapsed=%.2fs transformed_len=%d transformed_preview=%r",
            time.time() - t0,
            len(transformed_text),
            transformed_text[:120],
        )

        logger.info("process_stream:yield VOX")
        yield "VOX", source_text, transformed_text, None
        logger.info("process_stream:voice_context:start")
        voice_context = self._voice_context(audio_path, audio_key)
        logger.info(
            "process_stream:voice_context:done reference=%r ready=%s has_embedding=%s has_prosody=%s",
            voice_context.reference_audio_path,
            voice_context.ready,
            voice_context.speaker_embedding is not None,
            voice_context.prosody_profile is not None,
        )
        text_for_voice = transformed_text.strip() or source_text
        audio_emotion = self._collect_audio_emotion(audio_future, audio_executor)
        voice_output_prompt = self._build_voice_prompt(text_for_voice, audio_emotion, voice_output_prompt)
        voice_prompt_text = voice_output_prompt.instruction
        logger.info(
            "process_stream:voice:start text_for_voice_len=%d text_for_voice_preview=%r voice_prompt=%s",
            len(text_for_voice),
            text_for_voice[:120],
            "set" if voice_output_prompt is not None else "none",
        )
        cached_audio = None if self.disable_processing_cache else self.processing_cache.get_voice_output(text_for_voice, voice_prompt_text, voice_context)
        if cached_audio is not None and os.path.exists(cached_audio):
            output_audio_path = cached_audio
            logger.info("process_stream:voice:cache_hit output_audio_path=%s", output_audio_path)
        else:
            output_audio_path = self._step_voice(
                text_for_voice,
                audio_path,
                audio_key,
                voice_context,
                voice_output_prompt,
            )
            if not self.disable_processing_cache:
                self.processing_cache.set_voice_output(
                    text_for_voice,
                    voice_prompt_text,
                    voice_context,
                    output_audio_path,
                )
        logger.info(
            "process_stream:voice:done elapsed=%.2fs output_audio_path=%s",
            time.time() - t0,
            output_audio_path,
        )

        logger.info("process_stream:yield DONE")
        yield "DONE", source_text, transformed_text, output_audio_path
        if not self.disable_processing_cache:
            self.processing_cache.set_result(
                audio_key,
                text_transform_prompt.instruction,
                voice_prompt_text,
                PipelineResult(
                    output_audio_path=output_audio_path,
                    source_text=source_text,
                    transformed_text=transformed_text,
                    stt_latency=0.0,
                    text_transform_latency=0.0,
                    voice_output_latency=0.0,
                    total_latency=time.time() - t0,
                ),
            )

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
        voice_prompt_text = voice_output_prompt.instruction if voice_output_prompt is not None else ""
        cached_result = None
        if not self.disable_processing_cache and voice_output_prompt is not None:
            cached_result = self.processing_cache.get_result(audio_key, text_transform_prompt.instruction, voice_prompt_text)
        if cached_result is not None:
            logger.info("process_stream_live:full_cache_hit")
            self._emit(
                "cache.full_hit",
                "Full pipeline cache hit; STT/text/TTS were not executed.",
                "pipeline",
                cache="full_result",
                data={
                    "source_len": len(cached_result.source_text),
                    "transformed_len": len(cached_result.transformed_text),
                    "output_audio_path": cached_result.output_audio_path,
                },
            )
            yield "STT", cached_result.source_text, None, None
            yield "TXT", cached_result.source_text, cached_result.transformed_text, None
            yield "VOX", cached_result.source_text, cached_result.transformed_text, None
            yield "DONE", cached_result.source_text, cached_result.transformed_text, cached_result.output_audio_path
            return
        logger.info(
            "process_stream_live:audio_validated cache_key path=%s size=%d mtime=%.6f audio_key=%s",
            cache_key.path,
            cache_key.size,
            cache_key.mtime,
            audio_key,
        )
        audio_future, audio_executor = self._detect_audio_emotion_async(audio_path)

        # Check cache first
        cached_stt = None if self.disable_processing_cache else self.processing_cache.get_stt(audio_key)
        if cached_stt is not None:
            logger.info(
                "process_stream_live:stt_cache_hit source_len=%d source_preview=%r",
                len(cached_stt),
                cached_stt[:120],
            )
            self._emit(
                "cache.stt_hit",
                "STT cache hit; backend was not executed.",
                "stt",
                cache="stt",
                data={"source_len": len(cached_stt), "source_preview": cached_stt[:80]},
            )
            source_text = cached_stt
            logger.info("process_stream_live:yield STT")
            yield "STT", source_text, None, None
        else:
            self._emit(
                "stt.start",
                "STT started.",
                "stt",
                backend=type(self.stt_backend).__name__,
            )
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
                if not self.disable_processing_cache:
                    self.processing_cache.set_stt(audio_key, source_text)
                self._emit(
                    "stt.done",
                    "STT finished.",
                    "stt",
                    backend=type(self.stt_backend).__name__,
                    data={"source_len": len(source_text), "source_preview": source_text[:80]},
                )
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
                self._emit(
                    "stt.done",
                    "STT finished.",
                    "stt",
                    backend=type(self.stt_backend).__name__,
                    data={"source_len": len(source_text), "source_preview": source_text[:80]},
                )
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
        cached_text = None if self.disable_processing_cache else self.processing_cache.get_text(source_text, text_transform_prompt)
        if cached_text is not None:
            transformed_text = cached_text
            logger.info("process_stream_live:text:cache_hit transformed_len=%d", len(transformed_text))
        else:
            self._emit(
                "text.start",
                "Text transform started.",
                "text",
                backend=type(self.text_transform_backend).__name__,
            )
            transformed_text = self.text_transform_backend.transform(source_text, text_transform_prompt)
            if not validate_transform(source_text, transformed_text):
                self._emit("text_transform.validation_failed", "Text transform validation failed.", "text", level=EventLevel.WARNING, data={"source_len": len(source_text), "transformed_len": len(transformed_text)})
                transformed_text = PromptedRuleTextTransformBackend().transform(source_text, text_transform_prompt)
                self._emit("text_transform.fallback_used", "PromptedRuleTextTransformBackend fallback used.", "text")
            if not self.disable_processing_cache:
                self.processing_cache.set_text(source_text, text_transform_prompt, transformed_text)
            self._emit(
                "text.done",
                "Text transform finished.",
                "text",
                backend=type(self.text_transform_backend).__name__,
                data={"transformed_len": len(transformed_text), "transformed_preview": transformed_text[:80]},
            )
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
        voice_context = self._voice_context(audio_path, audio_key)
        logger.info(
            "process_stream_live:voice_context:done reference=%r ready=%s has_embedding=%s has_prosody=%s",
            voice_context.reference_audio_path,
            voice_context.ready,
            voice_context.speaker_embedding is not None,
            voice_context.prosody_profile is not None,
        )
        text_for_voice = transformed_text.strip() or source_text
        audio_emotion = self._collect_audio_emotion(audio_future, audio_executor)
        voice_output_prompt = self._build_voice_prompt(text_for_voice, audio_emotion, voice_output_prompt)
        voice_prompt_text = voice_output_prompt.instruction
        logger.info(
            "process_stream_live:voice:start text_for_voice_len=%d text_for_voice_preview=%r voice_prompt=%s",
            len(text_for_voice),
            text_for_voice[:120],
            "set" if voice_output_prompt is not None else "none",
        )
        cached_audio = None if self.disable_processing_cache else self.processing_cache.get_voice_output(text_for_voice, voice_prompt_text, voice_context)
        if cached_audio is not None and os.path.exists(cached_audio):
            output_audio_path = cached_audio
            logger.info("process_stream_live:voice:cache_hit output_audio_path=%s", output_audio_path)
        else:
            self._emit(
                "tts.start",
                "Voice output started.",
                "tts",
                backend=type(self.voice_output_backend).__name__,
            )
            output_audio_path = self._step_voice(
                text_for_voice,
                audio_path,
                audio_key,
                voice_context,
                voice_output_prompt,
            )
            if not self.disable_processing_cache:
                self.processing_cache.set_voice_output(
                    text_for_voice,
                    voice_prompt_text,
                    voice_context,
                    output_audio_path,
                )
            self._emit(
                "tts.done",
                "Voice output finished.",
                "tts",
                backend=type(self.voice_output_backend).__name__,
                data={"output_audio_path": output_audio_path},
            )
        logger.info(
            "process_stream_live:voice:done elapsed=%.2fs output_audio_path=%s",
            time.time() - t0,
            output_audio_path,
        )

        logger.info("process_stream_live:yield DONE")
        yield "DONE", source_text, transformed_text, output_audio_path
        if not self.disable_processing_cache:
            self.processing_cache.set_result(
                audio_key,
                text_transform_prompt.instruction,
                voice_prompt_text,
                PipelineResult(
                    output_audio_path=output_audio_path,
                    source_text=source_text,
                    transformed_text=transformed_text,
                    stt_latency=0.0,
                    text_transform_latency=0.0,
                    voice_output_latency=0.0,
                    total_latency=time.time() - t0,
                ),
            )
