"""Integration tests for real voice output backends.

These tests require irodori-tts or OpenVoice packages.
Skipped if not installed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import soundfile as sf

from kuchikae.domain.types import VoiceContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_reference_wav(path: str, sr: int = 24000, duration_sec: float = 2.0) -> None:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    signal = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), signal.astype(np.float32), sr)


# ---------------------------------------------------------------------------
# IrodoriTTSVoiceOutputBackend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not pytest.importorskip(
        "irodori_tts", reason="irodori-tts not installed"
    ),
    reason="irodori-tts not installed",
)
class TestIrodoriTTSVoiceOutputBackend:
    @pytest.mark.slow
    def test_synthesize_returns_path(self, tmp_path) -> None:
        from kuchikae.domain.voice_output import IrodoriTTSVoiceOutputBackend

        ref = tmp_path / "ref.wav"
        _write_reference_wav(str(ref), sr=24000)
        vc = VoiceContext(reference_audio_path=str(ref), ready=True)

        backend = IrodoriTTSVoiceOutputBackend(
            num_steps=4,
        )
        result = backend.synthesize("こんにちは、テストです。", vc)
        assert isinstance(result, str)
        assert os.path.isfile(result)

    @pytest.mark.slow
    def test_synthesize_fallback_on_no_reference(self) -> None:
        from kuchikae.domain.voice_output import IrodoriTTSVoiceOutputBackend

        vc = VoiceContext(reference_audio_path="", ready=False)
        backend = IrodoriTTSVoiceOutputBackend()
        result = backend.synthesize("テスト", vc)
        assert isinstance(result, str)

    @pytest.mark.slow
    def test_synthesize_empty_text_fallback(self, tmp_path) -> None:
        from kuchikae.domain.voice_output import IrodoriTTSVoiceOutputBackend

        ref = tmp_path / "ref.wav"
        _write_reference_wav(str(ref), sr=24000)
        vc = VoiceContext(reference_audio_path=str(ref), ready=True)
        backend = IrodoriTTSVoiceOutputBackend(num_steps=4)
        result = backend.synthesize("", vc)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# OpenVoiceOutputBackend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENVOICE_READY"),
    reason="OpenVoice not configured (set OPENVOICE_READY=1)",
)
class TestOpenVoiceOutputBackend:
    @pytest.mark.slow
    def test_synthesize_returns_path(self, tmp_path) -> None:
        from kuchikae.domain.voice_output import OpenVoiceOutputBackend

        ref = tmp_path / "ref.wav"
        _write_reference_wav(str(ref))
        vc = VoiceContext(reference_audio_path=str(ref), ready=True)

        backend = OpenVoiceOutputBackend()
        result = backend.synthesize("Hello, this is a test.", vc)
        assert isinstance(result, str)
        assert os.path.isfile(result)

    def test_fallback_on_no_reference(self) -> None:
        from kuchikae.domain.voice_output import OpenVoiceOutputBackend

        vc = VoiceContext(reference_audio_path="", ready=False)
        backend = OpenVoiceOutputBackend()
        result = backend.synthesize("test", vc)
        assert isinstance(result, str)

    def test_fallback_on_non_ready(self) -> None:
        from kuchikae.domain.voice_output import OpenVoiceOutputBackend

        vc = VoiceContext(reference_audio_path="/tmp/test.wav", ready=False)
        backend = OpenVoiceOutputBackend()
        result = backend.synthesize("test", vc)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Latency logging with real voice output
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_pipeline_latency_with_dummy_voice_logs_reports(tmp_path) -> None:
    from kuchikae.domain.metrics import LatencyLogger
    from kuchikae.domain.types import TextTransformPrompt
    from kuchikae.pipeline import KuchikaePipeline

    wav = tmp_path / "test.wav"
    sr = 16000
    sf.write(str(wav), np.zeros(sr, dtype=np.float32), sr)

    logger = LatencyLogger(log_dir=str(tmp_path / "logs"))
    pipeline = KuchikaePipeline(latency_logger=logger)
    prompt = TextTransformPrompt(instruction="polite")
    result = pipeline.process(str(wav), prompt)

    reports = logger.read_reports()
    assert len(reports) == 1
    assert "stt" in (reports[0].stages or {})
    assert "voice_output" in (reports[0].stages or {})

    assert result.stt_latency >= 0
    assert result.voice_output_latency >= 0
    assert result.total_latency > 0
