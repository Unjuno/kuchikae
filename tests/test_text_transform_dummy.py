"""Tests for DummyTextTransformBackend."""

import tempfile

from kuchikae.text_transform import DummyTextTransformBackend
from kuchikae.types import TextTransformPrompt


def test_dummy_transformer_returns_non_empty():
    backend = DummyTextTransformBackend()
    prompt = TextTransformPrompt(prompt_text="丁寧にして")
    result = backend.transform("こんにちは", prompt)

    assert isinstance(result, str)
    assert len(result) > 0


def test_dummy_transform_with_prompt_text():
    backend = DummyTextTransformBackend()
    prompt = TextTransformPrompt(prompt_text="カジュアルに")
    result = backend.transform("テスト", prompt)
    assert "テスト" in result
