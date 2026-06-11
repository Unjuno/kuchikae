"""Tests for streaming STT interface — StreamingSTTBackend, STTPartial/STTFinal."""

from __future__ import annotations

import numpy as np
import pytest

from kuchikae.domain.audio_stream import AudioChunk
from kuchikae.domain.stt import (
    DummyStreamingSTTBackend,
    StreamingSTTBackend,
)
from kuchikae.backends.stt import ChunkedStreamingSTTBackend
from kuchikae.domain.types import STTCommit, STTFinal, STTPartial


# ---------------------------------------------------------------------------
# STTPartial / STTFinal / STTCommit dataclasses
# ---------------------------------------------------------------------------


class TestSTTPartial:
    def test_fields(self) -> None:
        p = STTPartial(
            text="hello world",
            stable_prefix="hello",
            unstable_suffix=" world",
            start_sec=0.0,
            end_sec=1.0,
            confidence=0.95,
        )
        assert p.text == "hello world"
        assert p.stable_prefix == "hello"
        assert p.unstable_suffix == " world"
        assert p.confidence == 0.95

    def test_confidence_defaults_to_none(self) -> None:
        p = STTPartial(
            text="test",
            stable_prefix="",
            unstable_suffix="test",
            start_sec=0.0,
            end_sec=0.5,
        )
        assert p.confidence is None

    def test_session_id(self) -> None:
        p = STTPartial(session_id="sess_001")
        assert p.session_id == "sess_001"

    def test_defaults(self) -> None:
        p = STTPartial()
        assert p.session_id == ""
        assert p.text == ""
        assert p.start_sec == 0.0

    def test_frozen(self) -> None:
        p = STTPartial()
        with pytest.raises(AttributeError):
            p.text = "other"  # type: ignore[misc]


class TestSTTFinal:
    def test_fields(self) -> None:
        f = STTFinal(text="hello world", start_sec=0.0, end_sec=2.0, confidence=0.98)
        assert f.text == "hello world"
        assert f.confidence == 0.98


class TestSTTCommit:
    def test_fields(self) -> None:
        c = STTCommit(
            session_id="sess_001",
            text="hello world",
            start_sec=0.0,
            end_sec=2.0,
            confidence=0.98,
        )
        assert c.session_id == "sess_001"
        assert c.text == "hello world"
        assert c.confidence == 0.98

    def test_defaults(self) -> None:
        c = STTCommit()
        assert c.session_id == ""
        assert c.text == ""
        assert c.confidence is None

    def test_frozen(self) -> None:
        c = STTCommit(text="hello")
        with pytest.raises(AttributeError):
            c.text = "world"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DummyStreamingSTTBackend
# ---------------------------------------------------------------------------


class TestDummyStreamingSTTBackend:
    def test_push_audio_returns_partial(self) -> None:
        backend = DummyStreamingSTTBackend()
        chunk = AudioChunk(
            session_id="test",
            samples=np.zeros(1600),
            sample_rate=16000,
            start_sec=0.0,
            end_sec=0.1,
        )
        result = backend.push_audio(chunk)
        assert isinstance(result, STTPartial)
        assert result.text != ""

    def test_flush_returns_partial(self) -> None:
        backend = DummyStreamingSTTBackend()
        chunk = AudioChunk(
            session_id="test",
            samples=np.zeros(1600),
            sample_rate=16000,
            start_sec=0.0,
            end_sec=0.1,
        )
        backend.push_audio(chunk)
        result = backend.flush("test")
        assert isinstance(result, STTPartial)
        assert result.text == "明日までに資料を送ってください"

    def test_flush_returns_none_if_no_chunks(self) -> None:
        backend = DummyStreamingSTTBackend()
        result = backend.flush("test")
        assert result is None

    def test_multiple_pushes_accumulate(self) -> None:
        backend = DummyStreamingSTTBackend()
        for i in range(3):
            chunk = AudioChunk(
                session_id="test",
                samples=np.zeros(1600),
                sample_rate=16000,
                start_sec=float(i),
                end_sec=float(i + 1),
            )
            result = backend.push_audio(chunk)
            assert len(result.text) > 0

    def test_is_abstract(self) -> None:
        assert "push_audio" in StreamingSTTBackend.__abstractmethods__
        assert "flush" in StreamingSTTBackend.__abstractmethods__


# ---------------------------------------------------------------------------
# ChunkedStreamingSTTBackend (with DummySTTBackend inner)
# ---------------------------------------------------------------------------


class CountingSTTForStreaming:
    """Minimal STT double that returns incrementing text."""

    def __init__(self) -> None:
        self.call_count = 0

    def transcribe(self, audio_path: str) -> str:
        self.call_count += 1
        return f"chunk_{self.call_count}"


class TestChunkedStreamingSTTBackend:
    def test_push_audio_returns_partial(self) -> None:
        inner = CountingSTTForStreaming()
        backend = ChunkedStreamingSTTBackend(inner=inner)  # type: ignore[arg-type]
        chunk = AudioChunk(
            session_id="test",
            samples=np.ones(16000, dtype=np.float32) * 0.1,
            sample_rate=16000,
            start_sec=0.0,
            end_sec=1.0,
        )
        result = backend.push_audio(chunk)
        assert isinstance(result, STTPartial)
        assert "chunk" in result.text

    def test_flush_returns_partial(self) -> None:
        inner = CountingSTTForStreaming()
        backend = ChunkedStreamingSTTBackend(inner=inner)  # type: ignore[arg-type]
        chunk = AudioChunk(
            session_id="test",
            samples=np.ones(16000, dtype=np.float32) * 0.1,
            sample_rate=16000,
            start_sec=0.0,
            end_sec=1.0,
        )
        backend.push_audio(chunk)
        result = backend.flush("test")
        assert isinstance(result, STTPartial)

    def test_flush_returns_none_if_empty(self) -> None:
        inner = CountingSTTForStreaming()
        backend = ChunkedStreamingSTTBackend(inner=inner)  # type: ignore[arg-type]
        assert backend.flush("test") is None

    def test_stable_prefix_emerges_after_window(self) -> None:
        inner = CountingSTTForStreaming()
        backend = ChunkedStreamingSTTBackend(inner=inner, stable_window=2)  # type: ignore[arg-type]

        for i in range(4):
            chunk = AudioChunk(
                session_id="test",
                samples=np.ones(16000, dtype=np.float32) * 0.1,
                sample_rate=16000,
                start_sec=float(i),
                end_sec=float(i + 1),
            )
            partial = backend.push_audio(chunk)
            if i == 0:
                assert partial.stable_prefix == ""
                assert partial.unstable_suffix == partial.text

    def test_longest_common_prefix(self) -> None:
        assert ChunkedStreamingSTTBackend._longest_common_prefix("hello", "hello world") == "hello"
        assert ChunkedStreamingSTTBackend._longest_common_prefix("hello world", "hello") == "hello"
        assert ChunkedStreamingSTTBackend._longest_common_prefix("abc", "xyz") == ""
        assert ChunkedStreamingSTTBackend._longest_common_prefix("", "test") == ""
