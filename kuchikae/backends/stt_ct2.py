"""Specialized faster-whisper-compatible CT2-backed Japanese ASR models."""

from __future__ import annotations

from kuchikae.backends.stt import FasterWhisperSTTBackend


class AnimeWhisperCT2STTBackend(FasterWhisperSTTBackend):
    """Drop-in STT backend for faster-whisper-compatible anime-domain models.

    The current target model is `quantumcookie/anime-whisper-ct2`.
    We keep the implementation thin so benchmark output clearly reflects
    the load/transcribe behavior of the underlying model repo.
    """

    def __init__(self, model_size: str = "quantumcookie/anime-whisper-ct2-int8") -> None:
        super().__init__(model_size=model_size)


class AnimeWhisperCT2FP16STTBackend(AnimeWhisperCT2STTBackend):
    """Variant that targets the fp16 quantized CT2 repository."""

    def __init__(self) -> None:
        super().__init__(model_size="quantumcookie/anime-whisper-ct2-fp16")
