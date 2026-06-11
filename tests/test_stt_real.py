"""Integration tests for real STT backends.

These tests require the ``faster-whisper`` package.
Skipped if not installed.
"""

from __future__ import annotations

import pytest

from kuchikae.domain.stt import FasterWhisperSTTBackend


@pytest.fixture
def has_whisper():
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        return True
    except ImportError:
        return False


def _create_speech_wav(path: str, sr: int = 16000, duration_sec: float = 3.0) -> None:
    import numpy as np
    import soundfile as sf

    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # 440 Hz tone + harmonics to simulate speech-like energy
    signal = 0.1 * np.sin(2 * np.pi * 440 * t)
    signal += 0.05 * np.sin(2 * np.pi * 880 * t)
    signal += 0.03 * np.sin(2 * np.pi * 1320 * t)
    sf.write(str(path), signal.astype(np.float32), sr)


@pytest.mark.skipif(
    not pytest.importorskip("faster_whisper", reason="faster-whisper not installed"),
    reason="faster-whisper not installed",
)
class TestFasterWhisperSTTBackend:
    """Integration tests for FasterWhisperSTTBackend."""

    def test_transcribe_returns_string(self, tmp_path) -> None:
        wav = tmp_path / "speech.wav"
        _create_speech_wav(wav)
        backend = FasterWhisperSTTBackend(model_size="tiny")
        result = backend.transcribe(str(wav))
        assert isinstance(result, str)

    def test_transcribe_is_stable(self, tmp_path) -> None:
        wav = tmp_path / "stable.wav"
        _create_speech_wav(wav)
        backend = FasterWhisperSTTBackend(model_size="tiny")
        r1 = backend.transcribe(str(wav))
        r2 = backend.transcribe(str(wav))
        assert r1 == r2

    def test_transcribe_stream_yields_strings(self, tmp_path) -> None:
        wav = tmp_path / "stream_stt.wav"
        _create_speech_wav(wav, duration_sec=5.0)
        backend = FasterWhisperSTTBackend(model_size="tiny")
        results = list(backend.transcribe_stream(str(wav)))
        for r in results:
            assert isinstance(r, str)

    @pytest.mark.slow
    def test_transcribe_latency_report_with_logger(self, tmp_path) -> None:
        from kuchikae.domain.metrics import LatencyLogger
        from kuchikae.domain.types import TextTransformPrompt
        from kuchikae.pipeline import KuchikaePipeline

        wav = tmp_path / "latency_test.wav"
        _create_speech_wav(wav)
        logger = LatencyLogger(log_dir=str(tmp_path / "logs"))
        pipeline = KuchikaePipeline(
            stt_backend=FasterWhisperSTTBackend(model_size="tiny"),
            latency_logger=logger,
        )
        prompt = TextTransformPrompt(instruction="テスト")
        result = pipeline.process(str(wav), prompt)

        assert result.stt_latency > 0
        assert result.total_latency > 0

        reports = logger.read_reports()
        assert len(reports) == 1
        assert reports[0].stages is not None
        assert reports[0].stages["stt"] > 0
        assert reports[0].total_processing_sec > 0
        assert result.output_audio_path != ""
