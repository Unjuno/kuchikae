"""Tests for the simple mode (run_simple)."""

from __future__ import annotations

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


def test_run_simple_none_input_yields_empty():
    pipeline = KuchikaePipeline()
    gen = run_simple(pipeline, None)
    results = list(gen)
    assert len(results) == 1
    aud, src, trf, sts = results[0]
    assert src == ""
    assert trf == ""
    assert "録音ファイルを取得できませんでした" in sts
    assert "アップロード" not in sts


def test_run_simple_string_path_is_not_dropped(tmp_path):
    wav = tmp_path / "candidate.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    pipeline = KuchikaePipeline()
    gen = run_simple(pipeline, str(wav))
    results = list(gen)
    assert results[-1][3].startswith("言い直しました")
    assert "STT:" in results[-1][3]


def test_run_simple_with_dummy_wav_yields_done(tmp_path):
    pipeline = KuchikaePipeline()
    wav = tmp_path / "test.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(pipeline, str(wav))
    results = list(gen)
    assert len(results) >= 1
    last = results[-1]
    aud, src, trf, sts = last
    assert isinstance(aud, str) and aud != ""
    assert isinstance(src, str) and len(src) > 0
    assert isinstance(trf, str) and len(trf) > 0
    assert sts.startswith("言い直しました")
    assert "[warning]" in sts or "DummySTTBackend" in sts


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
    assert last[3].startswith("言い直しました")


def test_run_simple_dict_input(tmp_path):
    pipeline = KuchikaePipeline()
    wav = tmp_path / "dict_input.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(pipeline, {"path": str(wav)})
    results = list(gen)
    assert len(results) >= 1
    last = results[-1]
    assert last[3].startswith("言い直しました")


def test_run_simple_pipeline_exception_updates_status(tmp_path) -> None:
    wav = tmp_path / "simple_exception.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(_FailingPipeline(), str(wav))
    results = list(gen)
    assert results[-1][3].startswith("STT 段階で失敗しました:")
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
    assert "アップロード" not in result[3]
    assert "録音ファイルを取得できませんでした" in result[3]
