"""STTBackend and backends (Dummy + segmented wrapper)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Generator, List

import soundfile as sf

from kuchikae.domain.audio import AudioSegmenter, TranscriptJoiner
from kuchikae.domain.audio_stream import AudioChunk
from kuchikae.domain.types import STTPartial

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FasterWhisperConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "ja"
    beam_size: int = 1
    vad_filter: bool = True
    temperature: float = 0.0
    condition_on_previous_text: bool = False


STT_PRESETS: dict[str, FasterWhisperConfig] = {
    "fast": FasterWhisperConfig(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        beam_size=1,
        vad_filter=False,
    ),
    "balanced": FasterWhisperConfig(
        model_size="small",
        device="cpu",
        compute_type="int8",
        beam_size=1,
        vad_filter=True,
    ),
    "accurate": FasterWhisperConfig(
        model_size="medium",
        device="cpu",
        compute_type="int8",
        beam_size=3,
        vad_filter=True,
    ),
}


def resolve_stt_preset(name: str | None) -> FasterWhisperConfig:
    if name is None or name == "":
        return STT_PRESETS["balanced"]
    preset = STT_PRESETS.get(name)
    if preset is not None:
        return preset
    available = ", ".join(sorted(STT_PRESETS))
    raise ValueError(f"Unknown STT preset: {name!r}. Available presets: {available}")


class STTBackend:

    def transcribe(self, audio_path: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def transcribe_stream(self, audio_path: str) -> Generator[str, None, None]:
        """Optional: yield partial transcripts for streaming UI."""
        yield self.transcribe(audio_path)


class DummySTTBackend(STTBackend):

    def transcribe(self, audio_path: str) -> str:
        return "[DUMMY_STT_OUTPUT] 実音声は認識されていません"

    def transcribe_stream(self, audio_path: str) -> Generator[str, None, None]:
        yield "[DUMMY_STT_OUTPUT]"
        yield "[DUMMY_STT_OUTPUT] 実音声は"
        yield "[DUMMY_STT_OUTPUT] 実音声は認識されていません"


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
        self._final_text = "[DUMMY_STT_OUTPUT] 実音声は認識されていません"
        self._chunks = ["[DUMMY_STT_OUTPUT]", "実音声は認識されていません"]

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
            tmp_name = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_name = tmp.name
                sf.write(tmp_name, chunk.samples, chunk.sample_rate)
                text = self._inner.transcribe(tmp_name)
                results.append(text)
            finally:
                if tmp_name and os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        return self._joiner.join(results)

    def transcribe_stream(self, audio_path: str) -> Generator[str, None, None]:
        chunks = self._segmenter.segment(audio_path)
        logger.info("segmented stt stream: %d chunks", len(chunks))
        results: List[str] = []
        for chunk in chunks:
            tmp_name = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_name = tmp.name
                sf.write(tmp_name, chunk.samples, chunk.sample_rate)
                text = self._inner.transcribe(tmp_name)
                results.append(text)
                yield self._joiner.join(results)
            finally:
                if tmp_name and os.path.exists(tmp_name):
                    os.unlink(tmp_name)
