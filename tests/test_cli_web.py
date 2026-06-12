"""Tests for CLI and web module."""

from __future__ import annotations

import os
from unittest.mock import patch

from kuchikae.web import build_pipeline_config_from_env


def test_build_pipeline_config_dummy() -> None:
    config = build_pipeline_config_from_env(dummy=True)
    assert config["stt_backend"] == "dummy"
    assert config["text_transform_backend"] == "prompted_rule"
    assert config["voice_output_backend"] == "dummy"
    assert config["allow_dummy_backends"] is True


def test_build_pipeline_config_real() -> None:
    config = build_pipeline_config_from_env(real=True)
    assert config["stt_backend"] == "faster_whisper"
    assert config["text_transform_backend"] == "ollama"
    assert config["voice_output_backend"] == "irodori"
    assert config["allow_dummy_backends"] is False


def test_build_pipeline_config_default() -> None:
    config = build_pipeline_config_from_env()
    assert "stt_backend" in config
    assert "text_transform_backend" in config
    assert "voice_output_backend" in config


def test_build_pipeline_config_streaming() -> None:
    config = build_pipeline_config_from_env(dummy=True, streaming=True)
    assert config["streaming_stt"] is True


def test_cli_help(capsys) -> None:
    from kuchikae.cli import main
    with patch("sys.argv", ["kuchikae", "--help"]):
        main()
    captured = capsys.readouterr()
    assert "Usage:" in captured.out


def test_cli_doctor(capsys) -> None:
    from kuchikae.cli import main
    with patch("sys.argv", ["kuchikae", "doctor"]):
        main()
    captured = capsys.readouterr()
    assert "Kuchikae Doctor" in captured.out


def test_voice_style_presets_not_shadowed() -> None:
    import kuchikae.ui.handlers as handlers
    import kuchikae.domain.voice_style as voice_style
    assert handlers.VOICE_STYLE_PRESETS is voice_style.VOICE_STYLE_PRESETS
