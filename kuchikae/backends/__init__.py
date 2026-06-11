"""Real model backends for Kuchikae.

These backends depend on heavy ML libraries and are separated from the domain layer.
"""

from kuchikae.backends.stt import (
    ChunkedStreamingSTTBackend,
    FasterWhisperSTTBackend,
    StreamingFasterWhisperSTTBackend,
)

__all__ = [
    "ChunkedStreamingSTTBackend",
    "FasterWhisperSTTBackend",
    "StreamingFasterWhisperSTTBackend",
]
