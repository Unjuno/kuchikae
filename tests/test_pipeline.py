"""Tests for KuchikaePipeline — check_audio errors, cache behavior, streaming yields."""

from __future__ import annotations

import threading
import os
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from kuchikae.domain.audio_key import AudioKey
from kuchikae.domain.diagnostics import DiagnosticRecorder
from kuchikae.domain.audio_emotion import AudioEmotion
from kuchikae.counting_backends import (
    CountingSTTBackend,
    CountingTextTransformBackend,
    CountingVoiceContextExtractor,
    CountingVoiceOutputBackend,
)
from kuchikae.pipeline import KuchikaePipeline, MAX_AUDIO_DURATION, MAX_FILE_SIZE
from kuchikae.domain.types import TextTransformPrompt, VoiceContext, VoiceOutputPrompt


# ---------------------------------------------------------------------------
# check_audio
# ---------------------------------------------------------------------------

def _write_wav(path: str, duration_sec: float = 0.1, samplerate: int = 16000) -> None:
    data = np.zeros(int(samplerate * duration_sec), dtype=np.float32)
    sf.write(path, data, samplerate)


def test_check_audio_unsupported_extension(tmp_path) -> None:
    path = str(tmp_path / "test.txt")
    path_s = tmp_path / "test.txt"
    path_s.write_text("not audio")
    pipeline = KuchikaePipeline()
    with pytest.raises(ValueError, match="Unsupported format"):
        pipeline.check_audio(str(path_s))


def test_check_audio_nonexistent_file() -> None:
    pipeline = KuchikaePipeline()
    with pytest.raises(FileNotFoundError):
        pipeline.check_audio("/nonexistent/path.wav")


def test_check_audio_file_too_large(tmp_path) -> None:
    path = str(tmp_path / "large.wav")
    _write_wav(path)
    pipeline = KuchikaePipeline()
    with patch("os.path.getsize", return_value=MAX_FILE_SIZE + 1):
        with pytest.raises(ValueError, match="File too large"):
            pipeline.check_audio(path)


def test_check_audio_too_long(tmp_path) -> None:
    path = str(tmp_path / "long.wav")
    _write_wav(path)
    pipeline = KuchikaePipeline()
    with patch("soundfile.info") as mock_info:
        mock_info.return_value.duration = MAX_AUDIO_DURATION + 10
        with pytest.raises(ValueError, match="Audio too long"):
            pipeline.check_audio(path)


def test_check_audio_valid_wav_passes(tmp_path) -> None:
    path = str(tmp_path / "valid.wav")
    _write_wav(path)
    pipeline = KuchikaePipeline()
    assert pipeline.check_audio(path) is None


# ---------------------------------------------------------------------------
# _step_voice cache bug (pipeline.py:202 reads get_stt instead of get_voice_context)
# ---------------------------------------------------------------------------

def test_step_voice_does_not_overwrite_text_with_cached_stt(tmp_path) -> None:
    """_step_voice must NOT overwrite transformed text with cached STT result."""
    wav = tmp_path / "test.wav"
    _write_wav(str(wav))
    audio_key = AudioKey.from_file(str(wav))

    counting_vo = CountingVoiceOutputBackend()
    pipeline = KuchikaePipeline(voice_output_backend=counting_vo)

    vc = VoiceContext(reference_audio_path=str(wav), ready=True)

    pipeline.processing_cache.set_stt(audio_key, "CACHED STT RESULT")
    pipeline.processing_cache.set_voice_context(audio_key, vc)

    result = pipeline._step_voice(
        text="TRANSFORMED TEXT",
        audio_path=str(wav),
        audio_key=audio_key,
        voice_context=vc,
    )

    assert pipeline.voice_output_backend == counting_vo
    last_call_text = counting_vo.last_text
    assert last_call_text == "TRANSFORMED TEXT", (
        f"Expected 'TRANSFORMED TEXT', got {last_call_text!r}. "
        "Bug: _step_voice reads get_stt(audio_key) instead of get_voice_context."
    )


def test_step_voice_uses_cached_voice_context(tmp_path) -> None:
    wav = tmp_path / "test.wav"
    _write_wav(str(wav))
    audio_key = AudioKey.from_file(str(wav))

    counting_vc = CountingVoiceContextExtractor()
    pipeline = KuchikaePipeline()
    pipeline._voice_context_extractor = counting_vc

    cached_vc = VoiceContext(reference_audio_path="/cached/path.wav", ready=True)

    pipeline.processing_cache.set_voice_context(audio_key, cached_vc)

    pipeline._step_voice(
        text="hello",
        audio_path=str(wav),
        audio_key=audio_key,
        voice_context=VoiceContext(reference_audio_path="", ready=False),
    )

    assert counting_vc.call_count == 0, "Voice context extractor should NOT be called when cache hit"

    stored_vc = pipeline.processing_cache.get_voice_context(audio_key)
    assert stored_vc is not None
    assert stored_vc.reference_audio_path == "/cached/path.wav"


# ---------------------------------------------------------------------------
# process_stream
# ---------------------------------------------------------------------------

def test_process_stream_yields_correct_sequence(tmp_path) -> None:
    wav = tmp_path / "stream.wav"
    _write_wav(str(wav))
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="テスト")
    results = list(pipeline.process_stream(str(wav), prompt))

    assert len(results) >= 2

    statuses = [r[0] for r in results]
    assert statuses == ["STT", "TXT", "VOX", "DONE"]

    final = results[-1]
    assert final[0] == "DONE"
    assert isinstance(final[1], str) and len(final[1]) > 0
    assert isinstance(final[2], str) and len(final[2]) > 0
    assert isinstance(final[3], str) and len(final[3]) > 0


def test_process_stream_caches_stt_and_voice_context(tmp_path) -> None:
    wav = tmp_path / "cache.wav"
    _write_wav(str(wav))
    audio_key = AudioKey.from_file(str(wav))

    counting_stt = CountingSTTBackend()
    pipeline = KuchikaePipeline(stt_backend=counting_stt)
    prompt = TextTransformPrompt(instruction="テスト")

    list(pipeline.process_stream(str(wav), prompt))

    assert pipeline.processing_cache.get_stt(audio_key) is not None
    assert pipeline.processing_cache.get_voice_context(audio_key) is not None
    assert counting_stt.call_count == 1

    list(pipeline.process_stream(str(wav), prompt))
    assert counting_stt.call_count == 1, "STT should be called only once (cached)"


# ---------------------------------------------------------------------------
# process_stream_live
# ---------------------------------------------------------------------------

def test_process_stream_live_yields_correct_sequence(tmp_path) -> None:
    wav = tmp_path / "live.wav"
    _write_wav(str(wav))
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="テスト")
    results = list(pipeline.process_stream_live(str(wav), prompt))

    assert len(results) >= 2

    statuses = [r[0] for r in results]
    assert "STT" in statuses
    assert "TXT" in statuses
    assert "VOX" in statuses
    assert "DONE" in statuses

    final = results[-1]
    assert final[0] == "DONE"
    assert isinstance(final[1], str) and len(final[1]) > 0
    assert isinstance(final[2], str) and len(final[2]) > 0
    assert isinstance(final[3], str) and len(final[3]) > 0


def test_process_stream_live_uses_cached_stt(tmp_path) -> None:
    wav = tmp_path / "live_cache.wav"
    _write_wav(str(wav))
    audio_key = AudioKey.from_file(str(wav))

    counting_stt = CountingSTTBackend()
    pipeline = KuchikaePipeline(stt_backend=counting_stt)
    prompt = TextTransformPrompt(instruction="テスト")

    list(pipeline.process_stream_live(str(wav), prompt))
    assert counting_stt.call_count == 1

    list(pipeline.process_stream_live(str(wav), prompt))
    assert counting_stt.call_count == 1, "STT should be cached after first run"


def test_process_passes_voice_output_prompt(tmp_path) -> None:
    wav = tmp_path / "voice_prompt.wav"
    _write_wav(str(wav))

    counting_vo = CountingVoiceOutputBackend()
    pipeline = KuchikaePipeline(voice_output_backend=counting_vo)

    text_prompt = TextTransformPrompt(instruction="テスト")

    pipeline.process(str(wav), text_prompt, voice_style="auto")

    assert counting_vo.call_count == 1


def test_process_stream_passes_voice_output_prompt(tmp_path) -> None:
    wav = tmp_path / "voice_prompt_stream.wav"
    _write_wav(str(wav))

    counting_vo = CountingVoiceOutputBackend()
    pipeline = KuchikaePipeline(voice_output_backend=counting_vo)

    text_prompt = TextTransformPrompt(instruction="テスト")

    list(pipeline.process_stream(str(wav), text_prompt, voice_style="auto"))

    assert counting_vo.call_count == 1


def test_process_stream_live_passes_voice_output_prompt(tmp_path) -> None:
    wav = tmp_path / "voice_prompt_stream_live.wav"
    _write_wav(str(wav))

    counting_vo = CountingVoiceOutputBackend()
    pipeline = KuchikaePipeline(voice_output_backend=counting_vo)

    text_prompt = TextTransformPrompt(instruction="テスト")

    list(pipeline.process_stream_live(str(wav), text_prompt, voice_style="auto"))

    assert counting_vo.call_count == 1


def test_dummy_stt_sentinel_is_not_natural_sentence() -> None:
    from kuchikae.domain.stt import DummySTTBackend

    backend = DummySTTBackend()
    out = backend.transcribe("ignored.wav")
    assert out.startswith("[DUMMY_STT_OUTPUT]")
    assert "明日までに資料を送って" not in out


def test_create_pipeline_rejects_unknown_stt_backend_without_dummy() -> None:
    from kuchikae.pipeline import create_pipeline

    with pytest.raises(RuntimeError, match="Unknown or unavailable STT backend"):
        create_pipeline({"stt_backend": "definitely_not_real", "allow_dummy_backends": False})


def test_create_pipeline_uses_balanced_stt_preset_by_default() -> None:
    from kuchikae.pipeline import create_pipeline

    pipeline = create_pipeline({"allow_dummy_backends": True})
    assert getattr(pipeline, "stt_preset", None) == "balanced"


def test_resolve_stt_preset_returns_expected_default() -> None:
    from kuchikae.domain.stt import resolve_stt_preset

    preset = resolve_stt_preset("balanced")
    assert preset.model_size == "small"
    assert preset.device == "cpu"
    assert preset.compute_type == "int8"
    assert preset.beam_size == 1
    assert preset.vad_filter is True


def test_resolve_stt_presets_cover_fast_balanced_accurate() -> None:
    from kuchikae.domain.stt import resolve_stt_preset

    fast = resolve_stt_preset("fast")
    balanced = resolve_stt_preset("balanced")
    accurate = resolve_stt_preset("accurate")

    assert fast.model_size == "tiny"
    assert fast.device == "cpu"
    assert fast.compute_type == "int8"
    assert fast.beam_size == 1
    assert fast.vad_filter is False

    assert balanced.model_size == "small"
    assert balanced.beam_size == 1
    assert balanced.vad_filter is True

    assert accurate.model_size == "medium"
    assert accurate.beam_size == 3
    assert accurate.vad_filter is True


def test_resolve_stt_preset_rejects_unknown_name() -> None:
    from kuchikae.domain.stt import resolve_stt_preset

    with pytest.raises(ValueError, match="Available presets"):
        resolve_stt_preset("balnced")


def test_create_pipeline_passes_preset_to_faster_whisper_backend(monkeypatch) -> None:
    from kuchikae.pipeline import create_pipeline
    from kuchikae.backends import stt as stt_module

    captured = {}

    class FakeWhisperModel:
        pass

    class FakeFasterWhisperBackend:
        def __init__(self, config=None, **kwargs):
            captured["config"] = config
            self.config = config

    monkeypatch.setattr(stt_module, "FasterWhisperSTTBackend", FakeFasterWhisperBackend)
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", type("M", (), {"WhisperModel": FakeWhisperModel})())

    pipeline = create_pipeline({"allow_dummy_backends": True, "stt_preset": "fast"})
    assert getattr(pipeline, "stt_preset", None) == "fast"
    assert captured["config"].model_size == "tiny"
    assert captured["config"].vad_filter is False
    assert captured["config"].beam_size == 1


def test_disable_processing_cache_via_env(monkeypatch) -> None:
    from kuchikae.pipeline import create_pipeline

    monkeypatch.setenv("KUCHIKAE_DISABLE_PROCESSING_CACHE", "1")
    pipeline = create_pipeline({"allow_dummy_backends": True})

    assert getattr(pipeline, "disable_processing_cache", False) is True


class SpyProcessingCache:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_stt(self, *args, **kwargs):
        self.calls.append("get_stt")
        raise AssertionError("get_stt should not be called when cache is disabled")

    def set_stt(self, *args, **kwargs):
        self.calls.append("set_stt")
        raise AssertionError("set_stt should not be called when cache is disabled")

    def get_text(self, *args, **kwargs):
        self.calls.append("get_text")
        raise AssertionError("get_text should not be called when cache is disabled")

    def set_text(self, *args, **kwargs):
        self.calls.append("set_text")
        raise AssertionError("set_text should not be called when cache is disabled")

    def get_voice_context(self, *args, **kwargs):
        self.calls.append("get_voice_context")
        raise AssertionError("get_voice_context should not be called when cache is disabled")

    def set_voice_context(self, *args, **kwargs):
        self.calls.append("set_voice_context")
        raise AssertionError("set_voice_context should not be called when cache is disabled")

    def get_voice_output(self, *args, **kwargs):
        self.calls.append("get_voice_output")
        raise AssertionError("get_voice_output should not be called when cache is disabled")

    def set_voice_output(self, *args, **kwargs):
        self.calls.append("set_voice_output")
        raise AssertionError("set_voice_output should not be called when cache is disabled")

    def get_result(self, *args, **kwargs):
        self.calls.append("get_result")
        raise AssertionError("get_result should not be called when cache is disabled")

    def set_result(self, *args, **kwargs):
        self.calls.append("set_result")
        raise AssertionError("set_result should not be called when cache is disabled")


def test_disable_processing_cache_bypasses_all_cache_operations(tmp_path) -> None:
    wav = tmp_path / "nocache_all.wav"
    _write_wav(str(wav))
    spy_cache = SpyProcessingCache()
    counting_stt = CountingSTTBackend()
    counting_text = CountingTextTransformBackend()
    counting_vo = CountingVoiceOutputBackend()
    pipeline = KuchikaePipeline(
        stt_backend=counting_stt,
        text_transform_backend=counting_text,
        voice_output_backend=counting_vo,
        processing_cache=spy_cache,  # type: ignore[arg-type]
        disable_processing_cache=True,
    )
    prompt = TextTransformPrompt(instruction="テスト")

    list(pipeline.process_stream(str(wav), prompt))
    list(pipeline.process_stream_live(str(wav), prompt))
    pipeline.process(str(wav), prompt)

    assert spy_cache.calls == []
    assert counting_stt.call_count >= 3
    assert counting_text.call_count >= 3
    assert counting_vo.call_count >= 3


def test_set_stt_preset_rejects_unknown_without_mutating_state() -> None:
    pipeline = KuchikaePipeline(stt_preset="balanced", stt_config=None)
    original_preset = pipeline.stt_preset
    original_backend = pipeline.stt_backend

    with pytest.raises(ValueError):
        pipeline.set_stt_preset("balnced")

    assert pipeline.stt_preset == original_preset
    assert pipeline.stt_backend is original_backend


def test_processing_cache_can_be_disabled_via_config(tmp_path) -> None:
    wav = tmp_path / "nocache.wav"
    _write_wav(str(wav))
    audio_key = AudioKey.from_file(str(wav))
    counting_stt = CountingSTTBackend()
    pipeline = KuchikaePipeline(stt_backend=counting_stt, disable_processing_cache=True)
    prompt = TextTransformPrompt(instruction="テスト")

    list(pipeline.process_stream(str(wav), prompt))
    list(pipeline.process_stream(str(wav), prompt))

    assert counting_stt.call_count >= 2
    assert pipeline.processing_cache.get_stt(audio_key) is None


def test_stt_start_event_is_emitted(tmp_path) -> None:
    wav = tmp_path / "diag.wav"
    _write_wav(str(wav))
    diagnostics = DiagnosticRecorder(max_events=20)
    pipeline = KuchikaePipeline(diagnostics=diagnostics)
    pipeline.process(str(wav), TextTransformPrompt(instruction="テスト"))

    names = [event.name for event in diagnostics.events()]
    assert "stt.start" in names


def test_voice_prompt_auto_generation_and_cache_key(tmp_path) -> None:
    wav = tmp_path / "voice_auto.wav"
    _write_wav(str(wav))
    pipeline = KuchikaePipeline()
    result = pipeline.process(str(wav), TextTransformPrompt(instruction="テスト"), None)
    assert isinstance(result.output_audio_path, str)
    assert getattr(pipeline, "_last_voice_style", None) is not None


def test_audio_emotion_detector_can_timeout_without_failure(tmp_path, monkeypatch) -> None:
    class SlowDetector:
        def detect(self, audio_path: str):
            import time
            time.sleep(0.2)
            return AudioEmotion()

    wav = tmp_path / "slow_emotion.wav"
    _write_wav(str(wav))
    pipeline = KuchikaePipeline(audio_emotion_detector=SlowDetector(), voice_style_timeout_sec=0.01)
    result = pipeline.process(str(wav), TextTransformPrompt(instruction="テスト"), None)
    assert isinstance(result.output_audio_path, str)


@pytest.mark.parametrize("method_name", ["process", "process_stream", "process_stream_live"])
def test_audio_emotion_starts_before_stt_work(tmp_path, method_name) -> None:
    class BlockingDetector:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def detect(self, audio_path: str) -> AudioEmotion:
            self.started.set()
            self.release.wait(timeout=1.0)
            return AudioEmotion()

    class GuardedSTTBackend(CountingSTTBackend):
        def __init__(self, detector: BlockingDetector) -> None:
            super().__init__()
            self.detector = detector

        def transcribe(self, audio_path: str) -> str:
            assert self.detector.started.wait(timeout=0.5), "audio emotion should start before STT work"
            return super().transcribe(audio_path)

    wav = tmp_path / f"{method_name}.wav"
    _write_wav(str(wav))
    detector = BlockingDetector()
    pipeline = KuchikaePipeline(
        stt_backend=GuardedSTTBackend(detector),
        audio_emotion_detector=detector,
        voice_style_timeout_sec=0.01,
    )
    prompt = TextTransformPrompt(instruction="テスト")

    try:
        if method_name == "process":
            pipeline.process(str(wav), prompt, None)
        elif method_name == "process_stream":
            list(pipeline.process_stream(str(wav), prompt))
        else:
            list(pipeline.process_stream_live(str(wav), prompt))
    finally:
        detector.release.set()


def test_disabled_audio_emotion_detector_emits_no_start_event(tmp_path) -> None:
    from kuchikae.pipeline import create_pipeline

    wav = tmp_path / "disabled_audio_emotion.wav"
    _write_wav(str(wav))
    diagnostics = DiagnosticRecorder(max_events=50)
    pipeline = create_pipeline(
        {
            "allow_dummy_backends": True,
            "audio_emotion_detector": "disabled",
        }
    )
    pipeline.diagnostics = diagnostics
    pipeline.process(str(wav), TextTransformPrompt(instruction="テスト"), None)

    names = [event.name for event in diagnostics.events()]
    assert "audio_emotion.detect.start" not in names


def test_dummy_audio_emotion_detector_emits_start_and_done(tmp_path) -> None:
    from kuchikae.pipeline import create_pipeline

    wav = tmp_path / "dummy_audio_emotion.wav"
    _write_wav(str(wav))
    diagnostics = DiagnosticRecorder(max_events=50)
    pipeline = create_pipeline(
        {
            "allow_dummy_backends": True,
            "audio_emotion_detector": "dummy",
        }
    )
    pipeline.diagnostics = diagnostics
    pipeline.process(str(wav), TextTransformPrompt(instruction="テスト"), None)

    names = [event.name for event in diagnostics.events()]
    assert "audio_emotion.detect.start" in names
    assert "audio_emotion.detect.done" in names


# ---------------------------------------------------------------------------
# warmup — error handling should not raise
# ---------------------------------------------------------------------------

def test_warmup_noop_with_dummy_backends(tmp_path) -> None:
    pipeline = KuchikaePipeline()
    assert pipeline.warmup() is None


def test_warmup_does_not_raise_on_error(tmp_path) -> None:
    pipeline = KuchikaePipeline()

    with patch.object(pipeline.stt_backend, "transcribe", side_effect=Exception("boom")):
        pipeline.warmup()

    from unittest.mock import MagicMock
    mock_vo = MagicMock(spec=["_ensure_runtime"])
    mock_vo._ensure_runtime.side_effect = Exception("boom")
    old_vo = pipeline.voice_output_backend
    pipeline.voice_output_backend = mock_vo
    pipeline.warmup()
    pipeline.voice_output_backend = old_vo

    with patch("httpx.post", side_effect=Exception("boom")):
        pipeline.warmup()

    assert pipeline.warmup() is None  # survival check
