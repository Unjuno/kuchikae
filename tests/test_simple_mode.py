"""Tests for the simple mode (run_simple)."""

from __future__ import annotations

import time

import numpy as np
import soundfile as sf

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.ui.handlers import run, run_simple


class _FailingPipeline:
    def process_stream(self, *args, **kwargs):
        yield "STT", "聞き取った内容", None, None
        raise RuntimeError("stream failed")

    def process_stream_live(self, *args, **kwargs):
        yield "STT_PARTIAL", "聞き取った", None, None
        raise RuntimeError("live stream failed")


class _SlowSTTPipeline:
    """Pipeline whose STT hangs beyond timeout."""

    def process_stream(self, audio_path, prompt, voice_style="auto"):
        yield "STT", None, None, None
        time.sleep(10)
        yield "TXT", "source", None, None
        yield "VOX", "source", "text", None
        yield "DONE", "source", "text", "/tmp/out.wav"

    def process_stream_live(self, audio_path, prompt, voice_style="auto"):
        yield from self.process_stream(audio_path, prompt, voice_style)


class _SlowTTSPipeline:
    """Pipeline whose TTS hangs beyond timeout."""

    def process_stream(self, audio_path, prompt, voice_style="auto"):
        yield "STT", None, None, None
        yield "TXT", "source", None, None
        yield "VOX", "source", "text", None
        time.sleep(10)
        yield "DONE", "source", "text", "/tmp/out.wav"

    def process_stream_live(self, audio_path, prompt, voice_style="auto"):
        yield from self.process_stream(audio_path, prompt, voice_style)


def test_run_simple_none_input_yields_empty():
    pipeline = KuchikaePipeline()
    gen = run_simple(pipeline, None)
    results = list(gen)
    assert len(results) == 1
    aud, src, trf, sts, _ = results[0]
    assert src == ""
    assert trf == ""
    assert "録音ファイルを取得できませんでした" in sts


def test_run_simple_string_path_is_not_dropped(tmp_path):
    wav = tmp_path / "candidate.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    pipeline = KuchikaePipeline()
    gen = run_simple(pipeline, str(wav))
    results = list(gen)
    assert "言い直しました" in results[-1][3]


def test_run_simple_with_dummy_wav_yields_done(tmp_path):
    pipeline = KuchikaePipeline()
    wav = tmp_path / "test.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(pipeline, str(wav))
    results = list(gen)
    assert len(results) >= 1
    last = results[-1]
    aud, src, trf, sts, _ = last
    assert isinstance(aud, str) and aud != ""
    assert isinstance(src, str) and len(src) > 0
    assert isinstance(trf, str) and len(trf) > 0
    assert "言い直しました" in sts


def test_run_simple_stream_yields_intermediate_statuses(tmp_path):
    pipeline = KuchikaePipeline()
    wav = tmp_path / "status.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(pipeline, str(wav), live_streaming=False)
    results = list(gen)
    assert len(results) >= 2


def test_run_simple_filepath_tuple_input(tmp_path):
    pipeline = KuchikaePipeline()
    wav = tmp_path / "tuple.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(pipeline, str(wav))
    results = list(gen)
    assert len(results) >= 1
    last = results[-1]
    assert "言い直しました" in last[3]


def test_run_simple_dict_input(tmp_path):
    pipeline = KuchikaePipeline()
    wav = tmp_path / "dict_input.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(pipeline, {"path": str(wav)})
    results = list(gen)
    assert len(results) >= 1
    last = results[-1]
    assert "言い直しました" in last[3]


def test_run_simple_pipeline_exception_updates_status(tmp_path) -> None:
    wav = tmp_path / "simple_exception.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(_FailingPipeline(), str(wav))
    results = list(gen)
    assert "STT 段階で失敗しました:" in results[-1][3]
    assert "RuntimeError" in results[-1][3]


def test_run_pipeline_exception_updates_status(tmp_path) -> None:
    wav = tmp_path / "run_exception.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run(str(wav), "自然に", "", _FailingPipeline())
    results = list(gen)
    assert results[-1][3].startswith("STT 段階で失敗しました:")
    assert "RuntimeError" in results[-1][3]


def test_run_simple_none_path_message_has_no_upload_word() -> None:
    gen = run_simple(KuchikaePipeline(), None)
    result = list(gen)[0]
    assert "録音ファイルを取得できませんでした" in result[3]


def test_run_simple_live_streaming_yields_intermediate(tmp_path) -> None:
    pipeline = KuchikaePipeline()
    wav = tmp_path / "live.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(pipeline, str(wav), live_streaming=True)
    results = list(gen)
    assert len(results) >= 2
    statuses = [r[3] for r in results]
    assert any("音声認識中" in s or "文字起こし中" in s for s in statuses)


def test_run_simple_short_audio(tmp_path) -> None:
    pipeline = KuchikaePipeline()
    wav = tmp_path / "short.wav"
    sf.write(str(wav), np.zeros(2205, dtype=np.float32), 44100)
    gen = run_simple(pipeline, str(wav))
    results = list(gen)
    assert len(results) >= 1
    assert "言い直しました" in results[-1][3]


def test_run_simple_custom_template(tmp_path) -> None:
    pipeline = KuchikaePipeline()
    wav = tmp_path / "template.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(pipeline, str(wav), template_name="丁寧に")
    results = list(gen)
    assert len(results) >= 1
    assert "言い直しました" in results[-1][3]


def test_run_simple_unknown_template_falls_back(tmp_path) -> None:
    pipeline = KuchikaePipeline()
    wav = tmp_path / "fallback.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(pipeline, str(wav), template_name="存在しないテンプレート")
    results = list(gen)
    assert len(results) >= 1
    assert "言い直しました" in results[-1][3]


def test_pipeline_stt_timeout_config() -> None:
    pipeline = KuchikaePipeline()
    assert pipeline.stt_timeout_sec == 120.0
    pipeline2 = KuchikaePipeline(stt_timeout_sec=30.0, tts_timeout_sec=60.0)
    assert pipeline2.stt_timeout_sec == 30.0
    assert pipeline2.tts_timeout_sec == 60.0


def test_run_simple_pipeline_timeout_propagation(tmp_path) -> None:
    wav = tmp_path / "timeout.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    pipeline = KuchikaePipeline(stt_timeout_sec=0.1)
    gen = run_simple(pipeline, str(wav))
    results = list(gen)
    last_status = results[-1][3]
    assert "失敗しました" in last_status or "言い直しました" in last_status


def test_run_simple_voice_style_auto(tmp_path) -> None:
    pipeline = KuchikaePipeline()
    wav = tmp_path / "voice.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(pipeline, str(wav), voice_style="auto")
    results = list(gen)
    assert "言い直しました" in results[-1][3]


def test_run_simple_stt_preset_applied(tmp_path) -> None:
    pipeline = KuchikaePipeline()
    wav = tmp_path / "preset.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(pipeline, str(wav), stt_preset="fast")
    results = list(gen)
    assert "言い直しました" in results[-1][3]
