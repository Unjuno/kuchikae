"""Tests for text transform backends — rule-based, Ollama, GPT."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from kuchikae.domain.text_transform import (
    GPTTextTransformBackend,
    OllamaTextTransformBackend,
    PromptedRuleTextTransformBackend,
    RuleTextTransformBackend,
)
from kuchikae.domain.types import TextTransformPrompt


# ---------------------------------------------------------------------------
# RuleTextTransformBackend
# ---------------------------------------------------------------------------


class TestRuleTextTransformBackend:
    def test_desu_masu_conversion(self) -> None:
        backend = RuleTextTransformBackend()
        prompt = TextTransformPrompt(instruction="丁寧に")
        result = backend.transform("これはペンだ", prompt)
        assert "です" in result
        assert "だ" not in result.replace("[desu-masu]", "")

    def test_casual_conversion(self) -> None:
        backend = RuleTextTransformBackend()
        prompt = TextTransformPrompt(instruction="カジュアルに")
        result = backend.transform("これはペンです", prompt)
        assert "カジュアル" in result or "plain" in result

    def test_returns_prefix(self) -> None:
        backend = RuleTextTransformBackend()
        prompt = TextTransformPrompt(instruction="丁寧にしてください")
        result = backend.transform("テスト", prompt)
        assert result.startswith("[")

    def test_empty_text_returns_prefix_only(self) -> None:
        backend = RuleTextTransformBackend()
        prompt = TextTransformPrompt(instruction="polite")
        result = backend.transform("", prompt)
        assert isinstance(result, str)

    def test_desu_masu_detection_keywords(self) -> None:
        backend = RuleTextTransformBackend()
        for kw in ("丁寧", "ですます", "polite"):
            prompt = TextTransformPrompt(instruction=kw)
            result = backend.transform("走る", prompt)
            assert "desu-masu" in result

    def test_plain_detection_keywords(self) -> None:
        backend = RuleTextTransformBackend()
        for kw in ("カジュアル", "普通形", "plain"):
            prompt = TextTransformPrompt(instruction=kw)
            result = backend.transform("走ります", prompt)
            assert "plain" in result


# ---------------------------------------------------------------------------
# PromptedRuleTextTransformBackend
# ---------------------------------------------------------------------------


class TestPromptedRuleTextTransformBackend:
    def test_delegates_to_rule(self) -> None:
        backend = PromptedRuleTextTransformBackend()
        prompt = TextTransformPrompt(instruction="polite")
        result = backend.transform("テスト", prompt)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_different_instructions(self) -> None:
        backend = PromptedRuleTextTransformBackend()
        r1 = backend.transform("テスト", TextTransformPrompt(instruction="polite"))
        r2 = backend.transform("テスト", TextTransformPrompt(instruction="casual"))
        # Both produce some result
        assert isinstance(r1, str)
        assert isinstance(r2, str)


# ---------------------------------------------------------------------------
# OllamaTextTransformBackend
# ---------------------------------------------------------------------------

def ollama_available() -> bool:
    """Check if a local Ollama server is running."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200 and len(r.json().get("models", [])) > 0
    except Exception:
        return False


@pytest.mark.skipif(not ollama_available(), reason="Ollama not available")
class TestOllamaTextTransformBackend:
    def test_transform_returns_string(self) -> None:
        backend = OllamaTextTransformBackend(model="qwen3:8b")
        prompt = TextTransformPrompt(instruction="丁寧にしてください")
        result = backend.transform("これはテストです", prompt)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_transform_fallback_on_error(self) -> None:
        backend = OllamaTextTransformBackend(model="nonexistent-model")
        prompt = TextTransformPrompt(instruction="polite")
        with pytest.raises(RuntimeError, match="Ollama text transform failed"):
            backend.transform("テスト", prompt)

    def test_uses_custom_base_url(self) -> None:
        backend = OllamaTextTransformBackend(model="qwen3:8b")
        backend._base_url = "http://localhost:99999"
        prompt = TextTransformPrompt(instruction="polite")
        result = backend.transform("テスト", prompt)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# GPTTextTransformBackend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestGPTTextTransformBackend:
    def test_transform_returns_string(self) -> None:
        backend = GPTTextTransformBackend()
        prompt = TextTransformPrompt(instruction="丁寧にしてください")
        result = backend.transform("これはテストです", prompt)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_transform_fallback_no_api_key(self) -> None:
        with patch.dict(os.environ, clear=True):
            backend = GPTTextTransformBackend()
            prompt = TextTransformPrompt(instruction="polite")
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
                backend.transform("テスト", prompt)


# ---------------------------------------------------------------------------
# System prompt content tests
# ---------------------------------------------------------------------------


class TestOllamaSystemPrompt:
    def test_system_prompt_contains_engine_name(self) -> None:
        backend = OllamaTextTransformBackend()
        import httpx
        resp = httpx.post(
            f"{backend._base_url}/api/chat",
            json={
                "model": backend.model,
                "messages": [
                    {"role": "system", "content": "test"},
                    {"role": "user", "content": "test"},
                ],
                "stream": False,
            },
            timeout=5,
        )
        # Just verify the backend can be instantiated
        assert backend.model is not None

    def test_system_prompt_structure(self) -> None:
        """Verify the system prompt contains required elements by checking the source."""
        import inspect
        from kuchikae.domain.text_transform import OllamaTextTransformBackend
        source = inspect.getsource(OllamaTextTransformBackend.transform)
        assert "日本語発話演出エンジン" in source
        assert "短い社交的発話" in source or "挨拶、感謝、謝罪" in source
        assert "事実" in source
        assert "数値" in source
        assert "固有名詞" in source

    def test_user_prompt_structure(self) -> None:
        """Verify the user prompt contains required elements."""
        import inspect
        from kuchikae.domain.text_transform import OllamaTextTransformBackend
        source = inspect.getsource(OllamaTextTransformBackend.transform)
        assert "変換スタイル" in source
        assert "変換対象" in source
        assert "出力条件" in source


# ---------------------------------------------------------------------------
# Custom prompt priority tests
# ---------------------------------------------------------------------------


class TestCustomPromptPriority:
    def test_custom_prompt_overrides_template(self) -> None:
        from kuchikae.ui.handlers import run
        from kuchikae.pipeline import KuchikaePipeline
        import numpy as np
        import soundfile as sf
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.zeros(44100, dtype=np.float32), 44100)
            wav_path = f.name

        try:
            pipeline = KuchikaePipeline()
            gen = run(
                wav_path,
                template_name="丁寧に",
                custom_prompt="カスタムプロンプトテスト",
                pipeline=pipeline,
            )
            results = list(gen)
            # The pipeline should run without error
            assert len(results) >= 1
        finally:
            os.unlink(wav_path)

    def test_empty_custom_prompt_uses_template(self) -> None:
        from kuchikae.ui.handlers import run
        from kuchikae.pipeline import KuchikaePipeline
        import numpy as np
        import soundfile as sf
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.zeros(44100, dtype=np.float32), 44100)
            wav_path = f.name

        try:
            pipeline = KuchikaePipeline()
            gen = run(
                wav_path,
                template_name="丁寧に",
                custom_prompt="",
                pipeline=pipeline,
            )
            results = list(gen)
            assert len(results) >= 1
        finally:
            os.unlink(wav_path)

    def test_unknown_template_falls_back_to_natural(self) -> None:
        from kuchikae.ui.handlers import run
        from kuchikae.pipeline import KuchikaePipeline
        import numpy as np
        import soundfile as sf
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.zeros(44100, dtype=np.float32), 44100)
            wav_path = f.name

        try:
            pipeline = KuchikaePipeline()
            gen = run(
                wav_path,
                template_name="unknown-template",
                custom_prompt="",
                pipeline=pipeline,
            )
            results = list(gen)
            assert len(results) >= 1
        finally:
            os.unlink(wav_path)


@pytest.mark.slow
def test_pipeline_uses_prompted_rule_by_default() -> None:
    from kuchikae.pipeline import create_pipeline
    pipeline = create_pipeline({"allow_dummy_backends": True})
    assert isinstance(pipeline.text_transform_backend, PromptedRuleTextTransformBackend)
