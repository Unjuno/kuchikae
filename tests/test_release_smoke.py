"""Release smoke tests for CLI and web configuration."""

from __future__ import annotations

import sys


def test_build_pipeline_config_dummy_is_smoke_friendly() -> None:
    from kuchikae.web import build_pipeline_config_from_env

    config = build_pipeline_config_from_env(dummy=True)

    assert config["stt_backend"] == "dummy"
    assert config["text_transform_backend"] == "prompted_rule"
    assert config["voice_output_backend"] == "dummy"
    assert config["allow_dummy_backends"] is True
    assert config["streaming_stt"] is False


def test_build_pipeline_config_default_is_smoke_friendly(monkeypatch) -> None:
    from kuchikae.web import build_pipeline_config_from_env

    monkeypatch.delenv("KUCHIKAE_STT_BACKEND", raising=False)
    monkeypatch.delenv("KUCHIKAE_TEXT_BACKEND", raising=False)
    monkeypatch.delenv("KUCHIKAE_VOICE_BACKEND", raising=False)
    monkeypatch.delenv("KUCHIKAE_STREAMING_STT", raising=False)

    config = build_pipeline_config_from_env()

    assert config["stt_backend"] == "dummy"
    assert config["text_transform_backend"] == "prompted_rule"
    assert config["voice_output_backend"] == "dummy"
    assert config["allow_dummy_backends"] is True


def test_streaming_without_real_is_disabled() -> None:
    from kuchikae.web import build_pipeline_config_from_env

    config = build_pipeline_config_from_env(streaming=True)

    assert config["streaming_stt"] is False
    assert config["stt_backend"] == "dummy"


def test_streaming_with_real_is_enabled() -> None:
    from kuchikae.web import build_pipeline_config_from_env

    config = build_pipeline_config_from_env(real=True, streaming=True)

    assert config["streaming_stt"] is True
    assert config["stt_backend"] == "faster_whisper"


def test_cli_help_runs(capsys, monkeypatch) -> None:
    from kuchikae import cli

    monkeypatch.setattr(sys, "argv", ["kuchikae", "--help"])

    cli.main()

    out = capsys.readouterr().out
    assert "kuchikae serve" in out
    assert "kuchikae doctor" in out


def test_cli_doctor_runs_without_real_dependencies(capsys, monkeypatch) -> None:
    from kuchikae import cli

    monkeypatch.setattr(sys, "argv", ["kuchikae", "doctor"])

    cli.main()

    out = capsys.readouterr().out
    assert "Kuchikae Doctor" in out
    assert "Core dependencies" in out
    assert "Optional real backends" in out
