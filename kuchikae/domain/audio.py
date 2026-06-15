"""Audio chunking scaffold and shared audio utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
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
        if chunk_sec <= 0:
            raise ValueError("chunk_sec must be > 0")
        if overlap_sec < 0:
            raise ValueError("overlap_sec must be >= 0")
        if overlap_sec >= chunk_sec:
            raise ValueError("overlap_sec must be smaller than chunk_sec")
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


def linear_resample(samples: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.shape[0] / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32, copy=False)


@lru_cache(maxsize=1)
def torch_module():
    import torch
    return torch
