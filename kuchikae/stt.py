"""STTBackend and backends (Dummy + faster-whisper)."""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)


class STTBackend:

    def transcribe(self, audio_path: str) -> str:  # pragma: no cover
        raise NotImplementedError


class DummySTTBackend(STTBackend):

    def transcribe(self, audio_path: str) -> str:
        return "明日までに資料を送って"


class FasterWhisperSTTBackend(STTBackend):

    def __init__(self, model_size: str = "small") -> None:
        self._model_size = model_size
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "FasterWhisperSTTBackend requires the ``faster-whisper`` package. "
                "Install with ``uv pip install faster-whisper``."
            )

    def transcribe(self, audio_path: str) -> str:
        from faster_whisper import WhisperModel

        model_size = os.environ.get("WHISPER_MODEL_SIZE", self._model_size)
        logger.info("loading whisper model '%s' (CPU int8)...", model_size)
        t0 = time.time()
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info("whisper model loaded in %.2fs", time.time() - t0)

        t1 = time.time()
        segments, info = model.transcribe(audio_path, language="ja")
        logger.info("whisper transcribe: %.2fs", time.time() - t1)

        result = " ".join(seg.text for seg in segments)
        logger.info("whisper result: %s", result[:80])
        return result
