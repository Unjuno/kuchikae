"""Real model backends for Kuchikae.

These backends depend on heavy ML libraries and are separated from the domain layer.
"""

from kuchikae.backends.stt import (
    ChunkedStreamingSTTBackend,
    FasterWhisperSTTBackend,
    StreamingFasterWhisperSTTBackend,
)
from kuchikae.backends.stt_ct2 import AnimeWhisperCT2FP16STTBackend, AnimeWhisperCT2STTBackend
from kuchikae.backends.stt_nemo import ReazonSpeechNemoASRBackend
from kuchikae.backends.stt_transformers import TransformersJapaneseASRBackend
from kuchikae.backends.stt_transformers_whisper import TransformersWhisperJapaneseASRBackend
from kuchikae.backends.voice_output import (
    IrodoriTTSVoiceOutputBackend,
    OpenVoiceOutputBackend,
)

__all__ = [
    "ChunkedStreamingSTTBackend",
    "AnimeWhisperCT2FP16STTBackend",
    "AnimeWhisperCT2STTBackend",
    "FasterWhisperSTTBackend",
    "IrodoriTTSVoiceOutputBackend",
    "OpenVoiceOutputBackend",
    "ReazonSpeechNemoASRBackend",
    "TransformersJapaneseASRBackend",
    "TransformersWhisperJapaneseASRBackend",
    "StreamingFasterWhisperSTTBackend",
]
