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

