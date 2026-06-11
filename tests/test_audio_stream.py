"""Tests for streaming audio utilities — AudioChunker, EnergyVAD, AudioStreamBuffer."""

from __future__ import annotations

import numpy as np
import pytest

from kuchikae.domain.audio_stream import (
    AudioChunk,
    AudioChunker,
    AudioStreamBuffer,
    EnergyVAD,
)


# ---------------------------------------------------------------------------
# EnergyVAD
# ---------------------------------------------------------------------------


class TestEnergyVAD:
    def test_silence_is_not_speech(self) -> None:
        vad = EnergyVAD(threshold=0.02)
        silence = np.zeros(1600, dtype=np.float32)
        assert not vad.is_speech(silence)

    def test_loud_signal_is_speech(self) -> None:
        vad = EnergyVAD(threshold=0.02)
        loud = np.ones(1600, dtype=np.float32) * 0.5
        assert vad.is_speech(loud)

    def test_boundaries_single_utterance(self) -> None:
        vad = EnergyVAD(threshold=0.02)
        sr = 16000
        audio = np.zeros(sr * 3, dtype=np.float32)
        audio[int(0.5 * sr) : int(1.5 * sr)] = 0.3
        boundaries = vad.detect_boundaries(audio, sr)
        assert len(boundaries) == 1
        start, end = boundaries[0]
        assert 0.3 < start < 0.7
        assert 1.3 < end < 1.7

    def test_boundaries_two_utterances(self) -> None:
        vad = EnergyVAD(threshold=0.02, min_speech_frames=2)
        sr = 16000
        audio = np.zeros(sr * 4, dtype=np.float32)
        audio[int(0.3 * sr) : int(0.8 * sr)] = 0.3
        audio[int(2.0 * sr) : int(2.7 * sr)] = 0.3
        boundaries = vad.detect_boundaries(audio, sr)
        assert len(boundaries) == 2

    def test_boundaries_no_speech(self) -> None:
        vad = EnergyVAD(threshold=0.02)
        sr = 16000
        audio = np.zeros(sr * 2, dtype=np.float32)
        boundaries = vad.detect_boundaries(audio, sr)
        assert boundaries == []

    def test_rms_computation(self) -> None:
        samples = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
        expected = float(np.sqrt(np.mean(np.array([0.25, 0.25, 0.25, 0.25]))))
        from kuchikae.domain.audio_stream import _rms
        assert _rms(samples) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# AudioChunk
# ---------------------------------------------------------------------------


class TestAudioChunk:
    def test_fields(self) -> None:
        chunk = AudioChunk(
            samples=np.zeros(1600),
            sample_rate=16000,
            start_sec=0.0,
            end_sec=0.1,
            has_speech=True,
        )
        assert chunk.sample_rate == 16000
        assert chunk.start_sec == 0.0
        assert chunk.end_sec == 0.1
        assert chunk.has_speech


# ---------------------------------------------------------------------------
# AudioChunker
# ---------------------------------------------------------------------------


class TestAudioChunker:
    def test_push_single_chunk_emits_nothing_if_underfill(self) -> None:
        chunker = AudioChunker(sample_rate=16000, chunk_sec=2.0, hop_sec=1.0)
        samples = np.zeros(16000, dtype=np.float32)
        emitted = list(chunker.push(samples))
        assert emitted == []

    def test_push_emits_after_enough_data(self) -> None:
        chunker = AudioChunker(sample_rate=16000, chunk_sec=1.0, hop_sec=0.5)
        samples = np.zeros(32000, dtype=np.float32)
        emitted = list(chunker.push(samples))
        assert len(emitted) >= 1
        assert all(isinstance(c, AudioChunk) for c in emitted)

    def test_timestamps_are_monotonic(self) -> None:
        chunker = AudioChunker(sample_rate=16000, chunk_sec=0.5, hop_sec=0.25)
        samples = np.zeros(24000, dtype=np.float32)
        emitted = list(chunker.push(samples))
        for i in range(1, len(emitted)):
            assert emitted[i].start_sec > emitted[i - 1].start_sec

    def test_chunk_has_correct_length(self) -> None:
        chunker = AudioChunker(sample_rate=16000, chunk_sec=1.0, hop_sec=0.5)
        samples = np.zeros(32000, dtype=np.float32)
        emitted = list(chunker.push(samples))
        if emitted:
            assert len(emitted[0].samples) == 16000

    def test_flush_returns_remaining(self) -> None:
        chunker = AudioChunker(sample_rate=16000, chunk_sec=2.0, hop_sec=1.0)
        samples = np.zeros(16000, dtype=np.float32)
        list(chunker.push(samples))
        flushed = list(chunker.flush())
        assert len(flushed) == 1

    def test_stereo_is_downmixed(self) -> None:
        chunker = AudioChunker(sample_rate=16000, chunk_sec=0.5, hop_sec=0.25)
        stereo = np.zeros((16000, 2), dtype=np.float32)
        stereo[:, 0] = 0.1
        stereo[:, 1] = 0.2
        emitted = list(chunker.push(stereo))
        if emitted:
            assert emitted[0].samples.ndim == 1

    def test_reset_clears_buffer(self) -> None:
        chunker = AudioChunker(sample_rate=16000, chunk_sec=1.0, hop_sec=0.5)
        samples = np.zeros(32000, dtype=np.float32)
        list(chunker.push(samples))
        chunker.reset()
        assert chunker.buffered_sec == 0.0

    def test_speech_detection_in_chunk(self) -> None:
        chunker = AudioChunker(sample_rate=16000, chunk_sec=0.5, hop_sec=0.25)
        samples = np.ones(24000, dtype=np.float32) * 0.3
        emitted = list(chunker.push(samples))
        if emitted:
            assert emitted[0].has_speech

    def test_silence_detection_in_chunk(self) -> None:
        chunker = AudioChunker(sample_rate=16000, chunk_sec=0.5, hop_sec=0.25)
        samples = np.zeros(24000, dtype=np.float32)
        emitted = list(chunker.push(samples))
        if emitted:
            assert not emitted[0].has_speech


# ---------------------------------------------------------------------------
# AudioStreamBuffer
# ---------------------------------------------------------------------------


class TestAudioStreamBuffer:
    def test_accumulates_audio(self) -> None:
        buf = AudioStreamBuffer(session_id="sess_001", sample_rate=16000)
        chunk = AudioChunk(
            samples=np.ones(16000, dtype=np.float32),
            sample_rate=16000,
            start_sec=0.0,
            end_sec=1.0,
        )
        buf.push(chunk)
        assert buf.duration_sec == pytest.approx(1.0)
        assert len(buf.audio) == 16000

    def test_finalize_prevents_further_pushes(self) -> None:
        buf = AudioStreamBuffer(session_id="sess_002")
        buf.finalize()
        assert buf.is_finalized
        chunk = AudioChunk(
            samples=np.zeros(1600),
            sample_rate=16000,
            start_sec=0.0,
            end_sec=0.1,
        )
        with pytest.raises(RuntimeError, match="Cannot push to finalized"):
            buf.push(chunk)

    def test_multiple_chunks_concatenated(self) -> None:
        buf = AudioStreamBuffer(session_id="sess_003", sample_rate=16000)
        for i in range(3):
            chunk = AudioChunk(
                samples=np.ones(16000, dtype=np.float32) * i,
                sample_rate=16000,
                start_sec=float(i),
                end_sec=float(i + 1),
            )
            buf.push(chunk)
        assert buf.duration_sec == pytest.approx(3.0)
        assert np.allclose(buf.audio[:16000], 0)
        assert np.allclose(buf.audio[16000:32000], 1)

    def test_initial_state(self) -> None:
        buf = AudioStreamBuffer(session_id="sess_004")
        assert buf.duration_sec == 0.0
        assert not buf.is_finalized
        assert len(buf.audio) == 0
