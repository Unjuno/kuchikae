"""STTBackend and backends (Dummy + faster-whisper + segmented wrapper + streaming)."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Generator, List

import soundfile as sf

from kuchikae.audio import AudioSegmenter, TranscriptJoiner

logger = logging.getLogger(__name__)


class STTBackend:

    def transcribe(self, audio_path: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def transcribe_stream(self, audio_path: str) -> Generator[str, None, None]:
        """Optional: yield partial transcripts for streaming UI."""
        yield self.transcribe(audio_path)


class DummySTTBackend(STTBackend):

    def transcribe(self, audio_path: str) -> str:
        return "明日までに資料を送って"

    def transcribe_stream(self, audio_path: str) -> Generator[str, None, None]:
        yield "明日までに"
        yield "明日までに資料を"
        yield "明日までに資料を送って"


class FasterWhisperSTTBackend(STTBackend):

    def __init__(self, model_size: str = "small") -> None:
        self._model_size = model_size
        self._model = None
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "FasterWhisperSTTBackend requires the ``faster-whisper`` package. "
                "Install with ``uv pip install faster-whisper``."
            )

    def _load_model(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        model_size = os.environ.get("WHISPER_MODEL_SIZE", self._model_size)
        logger.info("loading whisper model '%s' (CPU int8)...", model_size)
        t0 = time.time()
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info("whisper model loaded in %.2fs", time.time() - t0)
        return self._model

    def transcribe(self, audio_path: str) -> str:
        model = self._load_model()

        t1 = time.time()
        segments, info = model.transcribe(audio_path, language="ja")
        logger.info("whisper transcribe: %.2fs", time.time() - t1)

        result = " ".join(seg.text for seg in segments)
        logger.info("whisper result: %s", result[:80])
        return result

    def transcribe_stream(self, audio_path: str) -> Generator[str, None, None]:
        model = self._load_model()

        t1 = time.time()
        segments, info = model.transcribe(audio_path, language="ja")
        logger.info("whisper transcribe: %.2fs", time.time() - t1)

        accumulated = []
        for seg in segments:
            accumulated.append(seg.text)
            yield " ".join(accumulated)


class StreamingFasterWhisperSTTBackend(FasterWhisperSTTBackend):
    """Streaming STT using fixed-window chunking for push-to-talk.
    
    Processes audio in small chunks and yields partial transcripts.
    Suitable for showing live transcription during/after recording.
    """

    def __init__(self, model_size: str = "small", chunk_sec: float = 5.0, overlap_sec: float = 1.0) -> None:
        super().__init__(model_size)
        self._chunk_sec = chunk_sec
        self._overlap_sec = overlap_sec

    def transcribe_stream(self, audio_path: str) -> Generator[str, None, None]:
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        chunk_samples = int(self._chunk_sec * sr)
        overlap_samples = int(self._overlap_sec * sr)
        stride = chunk_samples - overlap_samples
        total = len(data)
        
        model = self._load_model()
        accumulated = []
        
        start = 0
        while start < total:
            end = min(start + chunk_samples, total)
            chunk = data[start:end]
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, chunk, sr)
                segments, _ = model.transcribe(tmp.name, language="ja")
                os.unlink(tmp.name)
            
            chunk_text = " ".join(seg.text for seg in segments).strip()
            if chunk_text:
                accumulated.append(chunk_text)
                yield " ".join(accumulated)
            
            if end >= total:
                break
            start += stride


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
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, chunk.samples, chunk.sample_rate)
                text = self._inner.transcribe(tmp.name)
                results.append(text)
            os.unlink(tmp.name)
        return self._joiner.join(results)

    def transcribe_stream(self, audio_path: str) -> Generator[str, None, None]:
        chunks = self._segmenter.segment(audio_path)
        logger.info("segmented stt stream: %d chunks", len(chunks))
        results: List[str] = []
        for chunk in chunks:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, chunk.samples, chunk.sample_rate)
                text = self._inner.transcribe(tmp.name)
                results.append(text)
                yield self._joiner.join(results)
            os.unlink(tmp.name)
