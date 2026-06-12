from __future__ import annotations

from unittest.mock import patch

from kuchikae.domain.text_transform import OllamaTextTransformBackend, strip_cot, validate_transform
from kuchikae.domain.types import TextTransformPrompt


def test_strip_cot_removes_tags() -> None:
    assert strip_cot("a <think>b</think> c") == "a  c".strip()


def test_validate_transform_rejects_meta_and_numeric_loss() -> None:
    assert not validate_transform("今日は3件", "")
    assert not validate_transform("今日は3件", "理由: 今日は3件です")
    assert not validate_transform("今日は3件", "今日は")


def test_validate_transform_allows_short_kanji_only_japanese() -> None:
    assert validate_transform("ありがとう", "大丈夫")
    assert validate_transform("ありがとう", "了解")
    assert validate_transform("ありがとう", "承知")
    assert validate_transform("ありがとう", "感謝")
    assert validate_transform("ありがとう", "無理")
    assert validate_transform("ありがとう", "確認")
    assert validate_transform("ありがとう", "失礼")
    assert validate_transform("ありがとう", "了解です")
    assert validate_transform("ありがとう", "承知しました")


def test_validate_transform_rejects_cjk_only_long_output() -> None:
    assert not validate_transform("こんにちは", "你好，今天也请多关照")
    assert not validate_transform("ありがとう", "感谢您的帮助和支持非常感谢")


def test_validate_transform_allows_cjk_with_kana() -> None:
    assert validate_transform("こんにちは", "こんにちは、今日はいい天気ですね")


def test_ollama_transform_empty_result_returns_empty(monkeypatch) -> None:
    backend = OllamaTextTransformBackend(model="dummy", strict=False)

    class FakeResp:
        def raise_for_status(self):
            return None
        def json(self):
            return {"message": {"content": ""}}

    with patch("httpx.post", return_value=FakeResp()):
        out = backend.transform("hello", TextTransformPrompt(instruction="polite"))
    assert out == ""


def test_ollama_transform_validation_failed_returns_result(monkeypatch) -> None:
    backend = OllamaTextTransformBackend(model="dummy", strict=False)

    class FakeResp:
        def raise_for_status(self):
            return None
        def json(self):
            return {"message": {"content": "理由: テスト"}}

    with patch("httpx.post", return_value=FakeResp()):
        out = backend.transform("hello", TextTransformPrompt(instruction="polite"))
    assert out == "理由: テスト"


def test_ollama_transform_uses_content_not_thinking(monkeypatch) -> None:
    backend = OllamaTextTransformBackend(model="dummy", strict=False)

    class FakeResp:
        def raise_for_status(self):
            return None
        def json(self):
            return {"message": {"content": "ok", "thinking": "bad"}}

    with patch("httpx.post", return_value=FakeResp()):
        out = backend.transform("hello", TextTransformPrompt(instruction="polite"))
    assert out == "ok"

