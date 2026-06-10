"""Kuchikae pipeline orchestration."""

from __future__ import annotations

import os

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
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            has_faster_whisper = True
        except ImportError:
            has_faster_whisper = False
        stt = FasterWhisperSTTBackend() if has_faster_whisper else DummySTTBackend()

    text_backend_type = config.get("text_transform_backend", "ollama")
    text_backends = {
        "ollama": OllamaTextTransformBackend,
        "rule": RuleTextTransformBackend,
        "gpt_oss": GPTTextTransformBackend,
    }
    tt_class = text_backends.get(text_backend_type, OllamaTextTransformBackend)
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

    return KuchikaePipeline(
        stt_backend=stt,
        text_transform_backend=tt_class(),
        voice_output_backend=vo,
    )


class KuchikaePipeline:
    """Run the prompt-conditioned, voice-conditioned pipeline."""

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
    ) -> PipelineResult:
        """Process one utterance: STT → transform → voice output."""
        source_text = self.stt_backend.transcribe(audio_path)
        transformed_text = self.text_transform_backend.transform(source_text, text_transform_prompt)
        output_audio_path = self.voice_output_backend.synthesize(transformed_text, audio_path)
        return PipelineResult(output_audio_path=output_audio_path)
