"""Tests for the simple mode (run_simple)."""

from __future__ import annotations

import tempfile

import numpy as np
import soundfile as sf

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.ui import run_simple


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
    assert sts == "言い直しました"


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
    assert last[3] == "言い直しました"


def test_run_simple_dict_input(tmp_path):
    pipeline = KuchikaePipeline()
    wav = tmp_path / "dict_input.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100)
    gen = run_simple(pipeline, {"path": str(wav)})
    results = list(gen)
    assert len(results) >= 1
    last = results[-1]
    assert last[3] == "言い直しました"
