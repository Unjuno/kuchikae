"""Audio chunking scaffold."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
import soundfile as sf

from kuchikae.domain.types import AudioSegment


class AudioSegmenter(ABC):

    @abstractmethod
    def segment(self, audio_path: str) -> List[AudioSegment]:
        ...


class FixedWindowSegmenter(AudioSegmenter):

    def __init__(self, chunk_sec: float = 30.0, overlap_sec: float = 2.0) -> None:
        self._chunk_sec = chunk_sec
        self._overlap_sec = overlap_sec

    def segment(self, audio_path: str) -> List[AudioSegment]:
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        chunk_samples = int(self._chunk_sec * sr)
        overlap_samples = int(self._overlap_sec * sr)
        stride = chunk_samples - overlap_samples
        total = len(data)
        segments: List[AudioSegment] = []
        start = 0
        while start < total:
            end = min(start + chunk_samples, total)
            segments.append(AudioSegment(
                start_sec=start / sr,
                end_sec=end / sr,
                samples=data[start:end],
                sample_rate=sr,
            ))
            if end >= total:
                break
            start += stride
        return segments


class TranscriptJoiner:

    @staticmethod
    def join(transcripts: List[str]) -> str:
        return " ".join(t.strip() for t in transcripts if t.strip())
