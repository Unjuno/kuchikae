"""Kuchikae pipeline orchestration."""

from __future__ import annotations

import os
import time

from kuchikae.logging import setup_logger
from kuchikae.stt import (
    DummySTTBackend,
    FasterWhisperSTTBackend,
    STTBackend,
)
from kuchikae.text_transform import (
    DummyTextTransformBackend,
    GPTTextTransformBackend,
    OllamaTextTransformBackend,
    RuleTextTransformBackend,
    TextTransformBackend,
)
from kuchikae.types import (
    PipelineResult,
    TextTransformPrompt,
)
from kuchikae.voice_output import (
    DummyVoiceOutputBackend,
    IrodoriTTSVoiceOutputBackend,
    OpenVoiceOutputBackend,
    VoiceOutputBackend,
)

logger = setup_logger("kuchikae.pipeline")


def create_pipeline(backend_config: dict | None = None) -> KuchikaePipeline:
    config = backend_config or {}

    if config.get("stt_backend") == "faster_whisper":
        stt = FasterWhisperSTTBackend()
    else:
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            has_faster_whisper = True
        except ImportError:
            has_faster_whisper = False
        stt = FasterWhisperSTTBackend() if has_faster_whisper else DummySTTBackend()

    text_backend_type = config.get("text_transform_backend", "ollama")
    text_model = config.get("text_transform_model")
    text_backends = {
        "ollama": OllamaTextTransformBackend,
        "rule": RuleTextTransformBackend,
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
    ) -> None:
        self.stt_backend = stt_backend or DummySTTBackend()
        self.text_transform_backend = text_transform_backend or DummyTextTransformBackend()
        self.voice_output_backend = voice_output_backend or DummyVoiceOutputBackend()

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

    def process(
        self,
        audio_path: str,
        text_transform_prompt: TextTransformPrompt,
    ) -> PipelineResult:
        t0 = time.time()

        t1 = time.time()
        source_text = self.stt_backend.transcribe(audio_path)
        logger.info("STT: %.2fs → %s", time.time() - t1, source_text[:60])

        t2 = time.time()
        transformed_text = self.text_transform_backend.transform(source_text, text_transform_prompt)
        logger.info("TEXT: %.2fs → %s", time.time() - t2, transformed_text[:60])

        t3 = time.time()
        output_audio_path = self.voice_output_backend.synthesize(transformed_text, audio_path)
        logger.info("VOICE: %.2fs → %s", time.time() - t3, output_audio_path)

        total = time.time() - t0
        logger.info("TOTAL: %.2fs (STT %.2f + TEXT %.2f + VOICE %.2f)",
                    total, t2 - t1, t3 - t2, time.time() - t3)

        return PipelineResult(
            output_audio_path=output_audio_path,
            source_text=source_text,
            transformed_text=transformed_text,
        )
