"""Tests for incremental text transform — TransformState, TransformUpdate, IncrementalTextTransformBackend."""

from __future__ import annotations

import pytest

from kuchikae.domain.text_transform import (
    DummyIncrementalTextTransformBackend,
    IncrementalTextTransformBackend,
)
from kuchikae.domain.types import TextTransformPrompt, TransformState, TransformUpdate


# ---------------------------------------------------------------------------
# TransformState
# ---------------------------------------------------------------------------


class TestTransformState:
    def test_default_construction(self) -> None:
        s = TransformState()
        assert s.transformed_up_to == 0
        assert s.accumulated_output == ""

    def test_custom_state(self) -> None:
        s = TransformState(transformed_up_to=10, accumulated_output="hello world")
        assert s.transformed_up_to == 10
        assert s.accumulated_output == "hello world"


# ---------------------------------------------------------------------------
# TransformUpdate
# ---------------------------------------------------------------------------


class TestTransformUpdate:
    def test_fields(self) -> None:
        state = TransformState(transformed_up_to=5, accumulated_output="hello")
        update = TransformUpdate(new_output_segment=" world", updated_state=state)
        assert update.new_output_segment == " world"
        assert update.updated_state.transformed_up_to == 5


# ---------------------------------------------------------------------------
# DummyIncrementalTextTransformBackend
# ---------------------------------------------------------------------------


class TestDummyIncrementalTextTransformBackend:
    def test_transform_first_segment(self) -> None:
        backend = DummyIncrementalTextTransformBackend()
        prompt = TextTransformPrompt(instruction="")
        result = backend.transform_committed(
            "hello", TransformState(), prompt,
        )
        assert isinstance(result, TransformUpdate)
        assert "hello" in result.new_output_segment
        assert result.updated_state.transformed_up_to == 5

    def test_transform_second_segment(self) -> None:
        backend = DummyIncrementalTextTransformBackend()
        prompt = TextTransformPrompt(instruction="polite")
        prev = TransformState(transformed_up_to=5, accumulated_output="[transformed] hello")
        result = backend.transform_committed(
            "hello world", prev, prompt,
        )
        assert "world" in result.new_output_segment
        assert result.updated_state.transformed_up_to == 11

    def test_no_new_text_returns_empty(self) -> None:
        backend = DummyIncrementalTextTransformBackend()
        prompt = TextTransformPrompt(instruction="")
        prev = TransformState(transformed_up_to=5, accumulated_output="hello")
        result = backend.transform_committed("hello", prev, prompt)
        assert result.new_output_segment == ""
        assert result.updated_state is prev

    def test_is_abstract(self) -> None:
        assert "transform_committed" in IncrementalTextTransformBackend.__abstractmethods__

    def test_accumulated_output_grows(self) -> None:
        backend = DummyIncrementalTextTransformBackend()
        prompt = TextTransformPrompt(instruction="polite")
        state = TransformState()

        state = backend.transform_committed("hello", state, prompt).updated_state
        assert state.transformed_up_to == 5
        assert len(state.accumulated_output) > 0

        state = backend.transform_committed("hello world", state, prompt).updated_state
        assert state.transformed_up_to == 11
        assert "world" in state.accumulated_output

    def test_with_instruction_prefix(self) -> None:
        backend = DummyIncrementalTextTransformBackend()
        prompt = TextTransformPrompt(instruction="丁寧に")
        result = backend.transform_committed("テスト", TransformState(), prompt)
        assert "[transformed according to prompt]" in result.new_output_segment

    def test_without_instruction_prefix(self) -> None:
        backend = DummyIncrementalTextTransformBackend()
        prompt = TextTransformPrompt(instruction="")
        result = backend.transform_committed("テスト", TransformState(), prompt)
        assert "[transformed]" in result.new_output_segment
