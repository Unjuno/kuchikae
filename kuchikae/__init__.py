"""Kuchikae — prompt-conditioned voice transformation prototype."""

from kuchikae.audio_cache import AudioCache, VoiceContextExtractor
from kuchikae.audio_key import AudioKey
from kuchikae.pipeline import KuchikaePipeline
from kuchikae.stt import DummySTTBackend, STTBackend
from kuchikae.text_transform import (
    DummyTextTransformBackend,
    OllamaTextTransformBackend,
    TextTransformBackend,
)
from kuchikae.types import AudioCacheKey, PipelineResult, TextTransformPrompt, VoiceContext, VoiceOutputPrompt
from kuchikae.voice_output import (
    DummyVoiceOutputBackend,
    IrodoriTTSVoiceOutputBackend,
    VoiceOutputBackend,
)

__all__ = [
    "AudioCache",
    "AudioCacheKey",
    "AudioKey",
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
    "VoiceContext",
    "VoiceContextExtractor",
    "VoiceOutputBackend",
    "VoiceOutputPrompt",
]
