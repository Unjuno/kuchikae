"""STTBackend and backends (Dummy + faster-whisper + segmented wrapper)."""

from __future__ import annotations

import logging
import os
import time
from typing import List

from kuchikae.audio import AudioSegmenter, TranscriptJoiner

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


class SegmentedSTTBackend(STTBackend):

    def __init__(self, inner: STTBackend, segmenter: AudioSegmenter) -> None:
        self._inner = inner
        self._segmenter = segmenter
        self._joiner = TranscriptJoiner()

    def transcribe(self, audio_path: str) -> str:
        chunks = self._segmenter.segment(audio_path)
        logger.info("segmented stt: %d chunks", len(chunks))
        results: List[str] = []
        for chunk in chunks:
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, chunk.samples, chunk.sample_rate)
                text = self._inner.transcribe(tmp.name)
                results.append(text)
            os.unlink(tmp.name)
        return self._joiner.join(results)
