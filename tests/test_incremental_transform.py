"""Tests for incremental text transform — TransformUpdate, IncrementalTextTransformBackend."""

from __future__ import annotations

import pytest

from kuchikae.domain.text_transform import (
    DummyIncrementalTextTransformBackend,
    IncrementalTextTransformBackend,
)
from kuchikae.domain.types import (
    StreamingAudioSegment,
    TextTransformPrompt,
    TransformUpdate,
)


# ---------------------------------------------------------------------------
# TransformUpdate
# ---------------------------------------------------------------------------


class TestTransformUpdate:
    def test_fields(self) -> None:
        update = TransformUpdate(
            session_id="test",
            source_committed_text="hello world",
            transformed_committed_text="[transformed] hello world",
            newly_transformed_text="[transformed] hello world",
            is_final=False,
        )
        assert update.source_committed_text == "hello world"
        assert update.transformed_committed_text == "[transformed] hello world"
        assert update.newly_transformed_text == "[transformed] hello world"
        assert not update.is_final

    def test_defaults(self) -> None:
        update = TransformUpdate()
        assert update.session_id == ""
        assert update.source_committed_text == ""
        assert update.transformed_committed_text == ""
        assert update.newly_transformed_text == ""


# ---------------------------------------------------------------------------
# DummyIncrementalTextTransformBackend
# ---------------------------------------------------------------------------


class TestDummyIncrementalTextTransformBackend:
    def test_transform_first_segment(self) -> None:
        backend = DummyIncrementalTextTransformBackend()
        prompt = TextTransformPrompt(instruction="")
        result = backend.transform_committed(
            "hello", prompt, session_id="test",
        )
        assert isinstance(result, TransformUpdate)
        assert "hello" in result.newly_transformed_text
        assert result.transformed_committed_text == "[transformed] hello"

    def test_transform_second_segment(self) -> None:
        backend = DummyIncrementalTextTransformBackend()
        prompt = TextTransformPrompt(instruction="polite")
        backend.transform_committed("hello", TextTransformPrompt(instruction=""), session_id="test")
        result = backend.transform_committed(
            "hello world", prompt, session_id="test",
        )
        assert "world" in result.newly_transformed_text
        assert "world" in result.transformed_committed_text

    def test_no_new_text_returns_empty(self) -> None:
        backend = DummyIncrementalTextTransformBackend()
        prompt = TextTransformPrompt(instruction="")
        backend.transform_committed("hello", prompt, session_id="test")
        result = backend.transform_committed("hello", prompt, session_id="test")
        assert result.newly_transformed_text == ""
        assert result.transformed_committed_text == "[transformed] hello"

    def test_is_abstract(self) -> None:
        assert "transform_committed" in IncrementalTextTransformBackend.__abstractmethods__

    def test_accumulated_output_grows(self) -> None:
        backend = DummyIncrementalTextTransformBackend()
        prompt = TextTransformPrompt(instruction="polite")

        r1 = backend.transform_committed("hello", prompt, session_id="test")
        assert "hello" in r1.transformed_committed_text

        r2 = backend.transform_committed("hello world", prompt, session_id="test")
        assert "world" in r2.transformed_committed_text

    def test_with_instruction_prefix(self) -> None:
        backend = DummyIncrementalTextTransformBackend()
        prompt = TextTransformPrompt(instruction="丁寧に")
        result = backend.transform_committed("テスト", prompt, session_id="test")
        assert "[transformed according to prompt]" in result.newly_transformed_text

    def test_without_instruction_prefix(self) -> None:
        backend = DummyIncrementalTextTransformBackend()
        prompt = TextTransformPrompt(instruction="")
        result = backend.transform_committed("テスト", prompt, session_id="test")
        assert "[transformed]" in result.newly_transformed_text


# ---------------------------------------------------------------------------
# StreamingAudioSegment
# ---------------------------------------------------------------------------


class TestStreamingAudioSegment:
    def test_fields(self) -> None:
        seg = StreamingAudioSegment(
            session_id="sess_001",
            segment_index=2,
            text="hello",
            audio_path="out.wav",
            start_sec=0.0,
            end_sec=2.0,
            is_final=True,
        )
        assert seg.session_id == "sess_001"
        assert seg.segment_index == 2
        assert seg.text == "hello"
        assert seg.audio_path == "out.wav"
        assert seg.is_final

    def test_defaults(self) -> None:
        seg = StreamingAudioSegment()
        assert seg.session_id == ""
        assert seg.segment_index == 0
        assert seg.text == ""
        assert seg.audio_path is None
        assert not seg.is_final

    def test_frozen(self) -> None:
        seg = StreamingAudioSegment(text="hello")
        with pytest.raises(AttributeError):
            seg.text = "world"  # type: ignore[misc]

    def test_optional_timestamps(self) -> None:
        seg = StreamingAudioSegment(text="test")
        assert seg.start_sec is None
        assert seg.end_sec is None
