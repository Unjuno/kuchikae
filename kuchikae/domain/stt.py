"""STTBackend and backends (Dummy + segmented wrapper)."""

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
from kuchikae.domain.types import STTPartial

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


class StreamingSTTBackend(ABC):
    """Streaming STT backend interface.

    Push audio chunks incrementally; call ``flush()`` at the end.
    """

    @abstractmethod
    def push_audio(self, chunk: AudioChunk) -> STTPartial:
        ...

    @abstractmethod
    def flush(self, session_id: str) -> STTPartial | None:
        ...


class DummyStreamingSTTBackend(StreamingSTTBackend):
    """Dummy streaming STT for testing."""

    def __init__(self) -> None:
        self._pushed: dict[str, int] = {}
        self._final_text = "明日までに資料を送ってください"
        self._chunks = self._final_text.split("、")

    def push_audio(self, chunk: AudioChunk) -> STTPartial:
        sid = chunk.session_id
        pushed = self._pushed.get(sid, 0)
        frag = "".join(self._chunks[:pushed + 1])
        stable_prefix = "".join(self._chunks[:pushed])
        self._pushed[sid] = pushed + 1
        return STTPartial(
            session_id=sid,
            text=frag,
            stable_prefix=stable_prefix,
            unstable_suffix=frag[len(stable_prefix):],
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            confidence=0.95,
        )

    def flush(self, session_id: str) -> STTPartial | None:
        if self._pushed.get(session_id, 0) == 0:
            return None
        pushed = self._pushed[session_id]
        frag = "".join(self._chunks[:pushed])
        stable_prefix = "".join(self._chunks[:pushed - 1]) if pushed > 1 else ""
        return STTPartial(
            session_id=session_id,
            text=frag,
            stable_prefix=stable_prefix,
            unstable_suffix=frag[len(stable_prefix):],
            start_sec=0.0,
            end_sec=len(frag) * 0.1,
            confidence=0.95,
        )


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