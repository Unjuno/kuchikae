"""STTBackend and backends (Dummy + faster-whisper + segmented wrapper + streaming)."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Generator, List

import soundfile as sf

from kuchikae.domain.audio import AudioSegmenter, TranscriptJoiner
from kuchikae.domain.audio_stream import AudioChunk
from kuchikae.domain.types import STTFinal, STTPartial

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


class StreamingSTTBackend(ABC):
    """Streaming STT backend interface.

    Push audio chunks incrementally; call ``flush()`` at the end.
    """

    @abstractmethod
    def push_audio(self, chunk: AudioChunk) -> STTPartial:
        ...

    @abstractmethod
    def flush(self) -> STTFinal | None:
        ...


class DummyStreamingSTTBackend(StreamingSTTBackend):
    """Dummy streaming STT for testing."""

    def __init__(self) -> None:
        self._pushed = 0
        self._final_text = "明日までに資料を送ってください"
        self._chunks = self._final_text.split("、")

    def push_audio(self, chunk: AudioChunk) -> STTPartial:
        frag = "".join(self._chunks[:self._pushed + 1])
        stable_prefix = "".join(self._chunks[:self._pushed])
        self._pushed += 1
        return STTPartial(
            text=frag,
            stable_prefix=stable_prefix,
            unstable_suffix=frag[len(stable_prefix):],
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            confidence=0.95,
        )

    def flush(self) -> STTFinal | None:
        if self._pushed == 0:
            return None
        return STTFinal(
            text=self._final_text,
            start_sec=0.0,
            end_sec=len(self._final_text) * 0.1,
            confidence=0.95,
        )


class ChunkedStreamingSTTBackend(StreamingSTTBackend):
    """Real streaming STT that uses AudioChunker + FasterWhisperSTTBackend.

    Accumulates chunks and transcribes on each push.
    Uses simple stable-prefix heuristic: the text that hasn't changed
    across the last N iterations is considered stable.
    """

    def __init__(
        self,
        inner: FasterWhisperSTTBackend | None = None,
        stable_window: int = 2,
    ) -> None:
        self._inner = inner or FasterWhisperSTTBackend(model_size="tiny")
        self._stable_window = stable_window
        self._chunk_texts: list[str] = []

    def push_audio(self, chunk: AudioChunk) -> STTPartial:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, chunk.samples, chunk.sample_rate)
            text = self._inner.transcribe(tmp.name)
            os.unlink(tmp.name)

        self._chunk_texts.append(text)
        full_text = " ".join(self._chunk_texts)

        stable_prefix = ""
        last_n = self._chunk_texts[-self._stable_window:]
        if len(last_n) >= 2:
            if len(set(last_n)) == 1:
                stable_prefix = last_n[0]
            else:
                common = self._longest_common_prefix(last_n[0], last_n[-1])
                if len(common) > 2:
                    stable_prefix = common

        unstable_suffix = full_text[len(stable_prefix):] if stable_prefix else full_text

        return STTPartial(
            text=full_text,
            stable_prefix=stable_prefix,
            unstable_suffix=unstable_suffix,
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
        )

    def flush(self) -> STTFinal | None:
        if not self._chunk_texts:
            return None
        full = " ".join(self._chunk_texts)
        return STTFinal(text=full, start_sec=0.0, end_sec=0.0)

    @staticmethod
    def _longest_common_prefix(a: str, b: str) -> str:
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return a[:i]


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
