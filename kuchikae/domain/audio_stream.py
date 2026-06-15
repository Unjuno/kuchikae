"""Streaming audio utilities — chunker, VAD, session buffer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import numpy as np


# ---------------------------------------------------------------------------
# VAD — simple energy-based voice activity detection
# ---------------------------------------------------------------------------


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


class EnergyVAD:
    """Energy-based voice activity detector.

    Uses RMS threshold. Calibrated for 16-bit PCM normalized to [-1, 1].
    """

    def __init__(self, threshold: float = 0.02, min_speech_frames: int = 3) -> None:
        self.threshold = threshold
        self.min_speech_frames = min_speech_frames

    def is_speech(self, chunk: np.ndarray) -> bool:
        return _rms(chunk) > self.threshold

    def detect_boundaries(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_ms: int = 30,
        hop_ms: int = 10,
    ) -> list[tuple[float, float]]:
        frame_len = int(sample_rate * frame_ms / 1000)
        hop_len = int(sample_rate * hop_ms / 1000)
        total = len(audio)

        speech_frames: list[bool] = []
        pos = 0
        while pos < total:
            frame = audio[pos : pos + frame_len]
            speech_frames.append(self.is_speech(frame))
            pos += hop_len

        boundaries: list[tuple[float, float]] = []
        in_speech = False
        speech_start = 0.0
        consecutive_silence = 0

        for i, speaking in enumerate(speech_frames):
            t = i * hop_ms / 1000
            if speaking and not in_speech:
                speech_start = t
                in_speech = True
                consecutive_silence = 0
            elif not speaking and in_speech:
                consecutive_silence += 1
                if consecutive_silence >= self.min_speech_frames:
                    boundaries.append((speech_start, t))
                    in_speech = False
                    consecutive_silence = 0

        if in_speech:
            boundaries.append((speech_start, total / sample_rate))

        return boundaries


# ---------------------------------------------------------------------------
# AudioChunk — a single chunk of audio data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioChunk:
    session_id: str
    chunk_index: int = 0
    samples: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    sample_rate: int = 16000
    start_sec: float = 0.0
    end_sec: float = 0.0
    has_speech: bool = True
    is_final: bool = False

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be >= 0, got {self.chunk_index}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.start_sec < 0:
            raise ValueError(f"start_sec must be >= 0, got {self.start_sec}")
        if self.end_sec < self.start_sec:
            raise ValueError(
                f"end_sec ({self.end_sec}) must be >= start_sec ({self.start_sec})"
            )

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


# ---------------------------------------------------------------------------
# AudioChunker — accumulates PCM and emits fixed-size chunks
# ---------------------------------------------------------------------------


class AudioChunker:
    """Accumulates PCM audio chunks and emits fixed-size overlapping windows.

    Useful for streaming STT: emit chunks of *chunk_sec* seconds every *hop_sec*
    seconds while audio is arriving.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_sec: float = 2.0,
        hop_sec: float = 0.5,
        session_id: str = "default_session",
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_sec * sample_rate)
        self.hop_samples = int(hop_sec * sample_rate)
        self.session_id = session_id
        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._emitted_samples = 0
        self._chunk_index = 0

    @property
    def buffered_sec(self) -> float:
        return len(self._buffer) / self.sample_rate

    def push(self, samples: np.ndarray) -> Generator[AudioChunk, None, None]:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        self._buffer = np.concatenate([self._buffer, samples])

        while len(self._buffer) >= self.chunk_samples:
            chunk_data = self._buffer[: self.chunk_samples]
            self._buffer = self._buffer[self.hop_samples :]

            start_sec = self._emitted_samples / self.sample_rate
            end_sec = (self._emitted_samples + self.chunk_samples) / self.sample_rate

            rms_val = _rms(chunk_data)
            yield AudioChunk(
                session_id=self.session_id,
                chunk_index=self._chunk_index,
                samples=chunk_data,
                sample_rate=self.sample_rate,
                start_sec=start_sec,
                end_sec=end_sec,
                has_speech=rms_val > 0.01,
            )
            self._chunk_index += 1
            self._emitted_samples += self.hop_samples

    def flush(self) -> Generator[AudioChunk, None, None]:
        remaining = len(self._buffer)
        if remaining > 0:
            total_sec = self._emitted_samples / self.sample_rate
            yield AudioChunk(
                session_id=self.session_id,
                chunk_index=self._chunk_index,
                samples=self._buffer,
                sample_rate=self.sample_rate,
                start_sec=total_sec,
                end_sec=total_sec + remaining / self.sample_rate,
                has_speech=_rms(self._buffer) > 0.01,
                is_final=True,
            )
            self._chunk_index += 1
            self._buffer = np.array([], dtype=np.float32)

    def reset(self) -> None:
        self._buffer = np.array([], dtype=np.float32)
        self._emitted_samples = 0
        self._chunk_index = 0


# ---------------------------------------------------------------------------
# AudioStreamBuffer — session-level buffer
# ---------------------------------------------------------------------------


@dataclass
class AudioStreamBuffer:
    """Session-level audio buffer.

    Accumulates all audio for a session, tracks timing,
    and maintains a readable flag for streaming backends.
    """

    session_id: str
    sample_rate: int = 16000
    _accumulated: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    _start_time: float = 0.0
    _is_finalized: bool = False

    @property
    def duration_sec(self) -> float:
        return len(self._accumulated) / self.sample_rate

    @property
    def is_finalized(self) -> bool:
        return self._is_finalized

    @property
    def audio(self) -> np.ndarray:
        return self._accumulated

    def push(self, chunk: AudioChunk) -> None:
        if self._is_finalized:
            raise RuntimeError("Cannot push to finalized buffer")
        self._accumulated = np.concatenate([self._accumulated, chunk.samples])

    def finalize(self) -> None:
        self._is_finalized = True
