"""Tests for DummyTextTransformBackend."""

from __future__ import annotations

from kuchikae.domain.text_transform import DummyTextTransformBackend
from kuchikae.domain.types import TextTransformPrompt


def test_dummy_transformer_returns_non_empty():
    backend = DummyTextTransformBackend()
    prompt = TextTransformPrompt(instruction="丁寧にして")

    result = backend.transform("こんにちは", prompt)

    assert isinstance(result, str)
    assert len(result) > 0


def test_dummy_transform_preserves_source_text_in_scaffold_output():
    backend = DummyTextTransformBackend()
    prompt = TextTransformPrompt(instruction="カジュアルに")

    result = backend.transform("テスト", prompt)

    assert "テスト" in result
