"""STTBackend and backends (Dummy + faster-whisper)."""

from __future__ import annotations


class STTBackend:
    """Abstract base for speech-to-text backends."""

    def transcribe(self, audio_path: str) -> str:  # pragma: no cover
        raise NotImplementedError


class DummySTTBackend(STTBackend):
    """Returns a fixed Japanese sentence for v0.1."""

    def transcribe(self, audio_path: str) -> str:
        return "明日までに資料を送って"


class FasterWhisperSTTBackend(STTBackend):
    """Real STT backend using faster-whisper with the Japanese-small model (CPU).

    Model download is lazy — first call downloads if needed (~150 MB for small).
    Set environment variable ``WHISPER_MODEL_SIZE`` to control size: tiny, base,
    small, medium, large-v3 (default: small).
    """

    def __init__(self, model_size: str = "small") -> None:
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "FasterWhisperSTTBackend requires the ``faster-whisper`` package. "
                "Install with ``uv pip install faster-whisper``."
            )

    def transcribe(self, audio_path: str) -> str:
        from faster_whisper import WhisperModel

        model_size = os.environ.get("WHISPER_MODEL_SIZE", "small")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, language="ja")
        return " ".join(seg.text for seg in segments)
