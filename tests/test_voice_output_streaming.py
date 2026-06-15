"""Tests for streaming voice output — sentence segmenter, AudioSegmentQueue, StreamingVoiceOutputBackend."""

from __future__ import annotations

import os

import numpy as np

from kuchikae.domain.types import VoiceContext
from kuchikae.domain.voice_output import (
    AudioSegmentQueue,
    DummyStreamingVoiceOutputBackend,
    StreamingVoiceOutputBackend,
    segment_clauses,
    segment_sentences,
)


# ---------------------------------------------------------------------------
# Sentence / clause segmenter
# ---------------------------------------------------------------------------


class TestSentenceSegmenter:
    def test_single_sentence(self) -> None:
        result = segment_sentences("これはテストです。")
        assert result == ["これはテストです。"]

    def test_multiple_sentences(self) -> None:
        result = segment_sentences("これはテストです。次の文もあります。")
        assert result == ["これはテストです。", "次の文もあります。"]

    def test_mixed_punctuation(self) -> None:
        result = segment_sentences("本当ですか！確認してください？完了。")
        assert len(result) == 3

    def test_no_punctuation(self) -> None:
        result = segment_sentences("これはテストです")
        assert result == ["これはテストです"]

    def test_empty_text(self) -> None:
        assert segment_sentences("") == []
        assert segment_sentences("   ") == []


class TestClauseSegmenter:
    def test_single_clause(self) -> None:
        result = segment_clauses("これはテストです。")
        assert len(result) == 1

    def test_with_comma(self) -> None:
        result = segment_clauses("これは、テストです。")
        assert len(result) == 2

    def test_multiple_delimiters(self) -> None:
        result = segment_clauses("まず、最初に、確認します。次に、実行します。")
        assert len(result) == 5

    def test_no_delimiters(self) -> None:
        result = segment_clauses("テスト")
        assert result == ["テスト"]


# ---------------------------------------------------------------------------
# AudioSegmentQueue
# ---------------------------------------------------------------------------


class TestAudioSegmentQueue:
    def test_empty_queue(self) -> None:
        queue = AudioSegmentQueue()
        assert queue.count == 0
        merged = queue.merge()
        assert len(merged) == 0

    def test_single_segment(self) -> None:
        queue = AudioSegmentQueue()
        samples = np.ones(16000, dtype=np.float32)
        queue.enqueue(samples, 16000)
        assert queue.count == 1
        merged = queue.merge()
        assert len(merged) == 16000

    def test_multiple_segments_ordered(self) -> None:
        queue = AudioSegmentQueue()
        s1 = np.ones(16000, dtype=np.float32)
        s2 = np.ones(16000, dtype=np.float32) * 2
        queue.enqueue(s1, 16000)
        queue.enqueue(s2, 16000)
        merged = queue.merge(pause_sec=0.0)
        assert len(merged) == 32000
        assert np.allclose(merged[:16000], 1)
        assert np.allclose(merged[16000:], 2)

    def test_segments_out_of_order(self) -> None:
        queue = AudioSegmentQueue()

        class SeqBackend:
            def __init__(self):
                self._calls = []

            def synth(self, text: str) -> tuple[np.ndarray, int]:
                idx = int(text)
                samples = np.ones(1600, dtype=np.float32) * idx
                queue.enqueue(samples, 16000)
                return samples, 16000

        backend = SeqBackend()
        backend.synth("2")
        backend.synth("1")
        backend.synth("0")

        merged = queue.merge(pause_sec=0.0)
        assert len(merged) == 4800

    def test_clear(self) -> None:
        queue = AudioSegmentQueue()
        queue.enqueue(np.ones(1600), 16000)
        queue.clear()
        assert queue.count == 0
        assert queue.total_duration_sec == 0.0

    def test_pause_insertion(self) -> None:
        queue = AudioSegmentQueue()
        queue.enqueue(np.ones(16000), 16000)
        queue.enqueue(np.ones(16000), 16000)
        merged = queue.merge(pause_sec=0.5)
        expected_pause_samples = int(0.5 * 16000)
        assert len(merged) == 32000 + expected_pause_samples


# ---------------------------------------------------------------------------
# DummyStreamingVoiceOutputBackend
# ---------------------------------------------------------------------------


class TestDummyStreamingVoiceOutputBackend:
    def test_prepare_voice(self) -> None:
        backend = DummyStreamingVoiceOutputBackend()
        vc = VoiceContext(reference_audio_path="test.wav", ready=True)
        result = backend.prepare_voice(vc, session_id="test")
        assert result is None  # DummyVoiceOutputBackend.prepare_voice is a no-op

    def test_synthesize_segment_returns_audio(self) -> None:
        backend = DummyStreamingVoiceOutputBackend()
        samples, sr = backend.synthesize_segment("テスト", session_id="test")
        assert isinstance(samples, np.ndarray)
        assert sr > 0
        assert len(samples) > 0

    def test_finalize_returns_path(self) -> None:
        backend = DummyStreamingVoiceOutputBackend()
        vc = VoiceContext(reference_audio_path="test.wav", ready=True)
        backend.prepare_voice(vc, session_id="test")
        backend.synthesize_segment("最初の文。", session_id="test")
        backend.synthesize_segment("次の文。", session_id="test")
        path = backend.finalize(session_id="test")
        assert isinstance(path, str)
        assert os.path.isfile(path)

    def test_finalize_empty_returns_path(self) -> None:
        backend = DummyStreamingVoiceOutputBackend()
        vc = VoiceContext(reference_audio_path="test.wav", ready=True)
        backend.prepare_voice(vc, session_id="test2")
        path = backend.finalize(session_id="test2")
        assert isinstance(path, str)

    def test_is_abstract(self) -> None:
        assert "prepare_voice" in StreamingVoiceOutputBackend.__abstractmethods__
        assert "synthesize_segment" in StreamingVoiceOutputBackend.__abstractmethods__
        assert "finalize" in StreamingVoiceOutputBackend.__abstractmethods__

    def test_different_segments_produce_different_audio(self) -> None:
        backend = DummyStreamingVoiceOutputBackend()
        s1, _ = backend.synthesize_segment("短い", session_id="test")
        s2, _ = backend.synthesize_segment("これは長めのテスト文です", session_id="test")
        # Longer text should produce more samples
        assert len(s2) >= len(s1)

    def test_multiple_segments_with_pause(self) -> None:
        backend = DummyStreamingVoiceOutputBackend()
        vc = VoiceContext(reference_audio_path="test.wav", ready=True)
        backend.prepare_voice(vc, session_id="test")
        backend.synthesize_segment("一つ目。", session_id="test")
        backend.synthesize_segment("二つ目。", session_id="test")
        backend.synthesize_segment("三つ目。", session_id="test")
        path = backend.finalize(session_id="test")
        assert os.path.getsize(path) > 0
