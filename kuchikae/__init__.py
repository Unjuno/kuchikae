"""Kuchikae — prompt-conditioned voice transformation prototype."""

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.stt import DummySTTBackend, STTBackend
from kuchikae.text_transform import (
    DummyTextTransformBackend,
    OllamaTextTransformBackend,
    TextTransformBackend,
)
from kuchikae.types import PipelineResult, TextTransformPrompt
from kuchikae.voice_output import (
    DummyVoiceOutputBackend,
    IrodoriTTSVoiceOutputBackend,
    VoiceOutputBackend,
)

__all__ = [
    "DummySTTBackend",
    "DummyTextTransformBackend",
    "DummyVoiceOutputBackend",
    "IrodoriTTSVoiceOutputBackend",
    "KuchikaePipeline",
    "OllamaTextTransformBackend",
    "PipelineResult",
    "STTBackend",
    "TextTransformBackend",
    "TextTransformPrompt",
    "VoiceOutputBackend",
]
