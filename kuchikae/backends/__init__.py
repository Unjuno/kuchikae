"""Real model backends for Kuchikae.

These backends depend on heavy ML libraries and are separated from the domain layer.
"""

from kuchikae.backends.stt import (
    ChunkedStreamingSTTBackend,
    FasterWhisperSTTBackend,
    StreamingFasterWhisperSTTBackend,
)
from kuchikae.backends.voice_output import (
    IrodoriTTSVoiceOutputBackend,
    OpenVoiceOutputBackend,
)

__all__ = [
    "ChunkedStreamingSTTBackend",
    "FasterWhisperSTTBackend",
    "IrodoriTTSVoiceOutputBackend",
    "OpenVoiceOutputBackend",
    "StreamingFasterWhisperSTTBackend",
]
