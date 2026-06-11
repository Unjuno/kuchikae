"""Voice output backend interfaces."""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from kuchikae.domain.types import VoiceContext, VoiceOutputPrompt

logger = logging.getLogger(__name__)
OUTPUT_DIR = "outputs"


class VoiceOutputBackend(ABC):

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_context: VoiceContext,
        prompt: VoiceOutputPrompt | None = None,
    ) -> str:
        raise NotImplementedError


class DummyVoiceOutputBackend(VoiceOutputBackend):

    def synthesize(
        self,
        text: str,
        voice_context: VoiceContext,
        prompt: VoiceOutputPrompt | None = None,
    ) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "dummy.wav")
        duration_seconds = 1.0
        sample_rate = 44_100
        samples = np.zeros(int(duration_seconds * sample_rate), dtype=np.float32)
        sf.write(output_path, samples, sample_rate)
        return output_path


# ---------------------------------------------------------------------------
# Sentence / clause segmenter
# ---------------------------------------------------------------------------


def segment_sentences(text: str) -> list[str]:
    """Split text into sentences or clause-like units.

    Uses Japanese sentence-ending punctuation (。！？) and
    English sentence-ending punctuation (.!?) as delimiters.
    """
    parts = re.split(r"(?<=[。！？．.!?])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def segment_clauses(text: str) -> list[str]:
    """Split text into shorter clause-like units.

    In addition to sentence boundaries, splits on 読点 (、)
    and commas to produce shorter segments for faster TTS output.
    """
    parts = re.split(r"(?<=[。！？．.!?、，])\s*", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Audio segment queue — ordered merging of incremental TTS outputs
# ---------------------------------------------------------------------------


@dataclass
class QueuedAudioSegment:
    samples: np.ndarray
    sample_rate: int
    index: int


class AudioSegmentQueue:
    """Ordered queue for incremental TTS audio segments.

    Collects segments produced in any order and merges them
    in correct sequence with an optional pause between segments.
    """

    def __init__(self, pause_sec: float = 0.15) -> None:
        self._segments: list[QueuedAudioSegment] = []
        self._next_index = 0
        self._pause_samples = 0

    @property
    def total_duration_sec(self) -> float:
        total = 0.0
        for seg in self._segments:
            total += len(seg.samples) / seg.sample_rate
        return total

    def enqueue(self, samples: np.ndarray, sample_rate: int) -> int:
        idx = self._next_index
        self._segments.append(QueuedAudioSegment(
            samples=samples, sample_rate=sample_rate, index=idx,
        ))
        self._next_index += 1
        return idx

    def merge(self, pause_sec: float = 0.15) -> np.ndarray:
        if not self._segments:
            return np.array([], dtype=np.float32)

        self._segments.sort(key=lambda s: s.index)
        sr = self._segments[0].sample_rate
        pause = np.zeros(int(pause_sec * sr), dtype=np.float32)

        parts: list[np.ndarray] = []
        for i, seg in enumerate(self._segments):
            if i > 0:
                parts.append(pause)
            parts.append(seg.samples)

        return np.concatenate(parts)

    def clear(self) -> None:
        self._segments.clear()
        self._next_index = 0

    @property
    def count(self) -> int:
        return len(self._segments)


# ---------------------------------------------------------------------------
# Streaming voice output backend interface
# ---------------------------------------------------------------------------


class StreamingVoiceOutputBackend(ABC):
    """Incremental / streaming voice output backend.

    Prepare voice context once, then synthesize segments as they become
    available, and finalize to produce the complete output file.
    """

    @abstractmethod
    def prepare_voice(self, voice_context: VoiceContext, session_id: str = "") -> None:
        ...

    @abstractmethod
    def synthesize_segment(self, text_segment: str, session_id: str = "", segment_index: int = 0) -> tuple[np.ndarray, int]:
        ...

    @abstractmethod
    def finalize(self, session_id: str = "") -> str:
        ...


class DummyStreamingVoiceOutputBackend(StreamingVoiceOutputBackend):
    """Dummy streaming voice output for testing.

    Generates a short sine tone for each segment.
    Per-session state is maintained internally.
    """

    def __init__(self) -> None:
        self._queues: dict[str, AudioSegmentQueue] = {}
        self._segment_indices: dict[str, int] = {}

    def _queue(self, session_id: str) -> AudioSegmentQueue:
        if session_id not in self._queues:
            self._queues[session_id] = AudioSegmentQueue()
            self._segment_indices[session_id] = 0
        return self._queues[session_id]

    def prepare_voice(self, voice_context: VoiceContext, session_id: str = "") -> None:
        q = self._queue(session_id)
        q.clear()
        self._segment_indices[session_id] = 0

    def synthesize_segment(self, text_segment: str, session_id: str = "", segment_index: int = 0) -> tuple[np.ndarray, int]:
        sr = 16000
        duration = max(0.3, len(text_segment) * 0.05)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        samples = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        q = self._queue(session_id)
        q.enqueue(samples, sr)
        self._segment_indices[session_id] += 1
        return samples, sr

    def finalize(self, session_id: str = "") -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"streaming_output_{int(time.time())}.wav")
        q = self._queue(session_id)
        merged = q.merge()
        sf.write(path, merged, 16000)
        return path
