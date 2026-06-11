"""Tests for audio chunking scaffold."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from kuchikae.domain.audio import (
    FixedWindowSegmenter,
    TranscriptJoiner,
)
from kuchikae.domain.types import AudioSegment


def test_audio_segment_fields() -> None:
    seg = AudioSegment(start_sec=0.0, end_sec=1.0, samples=np.zeros(44100), sample_rate=44100)
    assert seg.start_sec == 0.0
    assert seg.end_sec == 1.0
    assert len(seg.samples) == 44100


def test_fixed_window_no_overlap(tmp_path) -> None:
    sr = 16000
    data = np.sin(2 * np.pi * 440 * np.arange(sr * 3) / sr, dtype=np.float32)
    wav = tmp_path / "test.wav"
    sf.write(str(wav), data, sr)

    segmenter = FixedWindowSegmenter(chunk_sec=1.0, overlap_sec=0.0)
    segs = segmenter.segment(str(wav))
    assert len(segs) == 3
    assert abs(segs[0].end_sec - segs[1].start_sec) < 0.01
    assert all(s.sample_rate == sr for s in segs)
    assert abs(segs[0].end_sec - segs[0].start_sec - 1.0) < 0.01


def test_fixed_window_with_overlap(tmp_path) -> None:
    sr = 16000
    data = np.zeros(sr * 4, dtype=np.float32)
    wav = tmp_path / "overlap.wav"
    sf.write(str(wav), data, sr)

    segmenter = FixedWindowSegmenter(chunk_sec=2.0, overlap_sec=0.5)
    segs = segmenter.segment(str(wav))
    assert len(segs) >= 2
    overlap = segs[0].end_sec - segs[1].start_sec
    assert abs(overlap - 0.5) < 0.01


def test_fixed_window_short_audio(tmp_path) -> None:
    sr = 16000
    data = np.zeros(sr, dtype=np.float32)
    wav = tmp_path / "short.wav"
    sf.write(str(wav), data, sr)

    segmenter = FixedWindowSegmenter(chunk_sec=10.0, overlap_sec=0.0)
    segs = segmenter.segment(str(wav))
    assert len(segs) == 1
    assert abs(segs[0].end_sec - 1.0) < 0.01


def test_fixed_window_stereo(tmp_path) -> None:
    sr = 16000
    data = np.zeros((sr * 2, 2), dtype=np.float32)
    wav = tmp_path / "stereo.wav"
    sf.write(str(wav), data, sr)

    segmenter = FixedWindowSegmenter(chunk_sec=1.0, overlap_sec=0.0)
    segs = segmenter.segment(str(wav))
    assert len(segs) == 2
    assert segs[0].samples.ndim == 1


def test_transcript_joiner_empty() -> None:
    assert TranscriptJoiner.join([]) == ""


def test_transcript_joiner_single() -> None:
    assert TranscriptJoiner.join(["hello"]) == "hello"


def test_transcript_joiner_multiple() -> None:
    result = TranscriptJoiner.join(["hello", "world", "foo"])
    assert result == "hello world foo"


def test_transcript_joiner_strips_whitespace() -> None:
    result = TranscriptJoiner.join(["  hello  ", "", "  world  "])
    assert result == "hello world"
