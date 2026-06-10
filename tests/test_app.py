"""Tests for audio input normalization."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from kuchikae.ui import normalize_audio_path


def test_str_existing_file(tmp_path: str) -> None:
    wav = tmp_path / "x.wav"
    sf.write(str(wav), np.zeros(44_100, dtype=np.float32), 44_100)
    assert normalize_audio_path(str(wav)) == str(wav)


def test_str_nonexistent_file(tmp_path: str) -> None:
    assert normalize_audio_path(str(tmp_path / "missing.wav")) is None


def test_none() -> None:
    assert normalize_audio_path(None) is None


def test_tuple_returns_tmp_path() -> None:
    sr = 48_000
    data = np.zeros(sr, dtype=np.float32)
    path = normalize_audio_path((sr, data))
    assert isinstance(path, str) and os.path.isfile(path)


def test_dict_with_orig_name(tmp_path: str) -> None:
    wav = tmp_path / "orig.wav"
    sf.write(str(wav), np.zeros(44_100, dtype=np.float32), 44_100)
    assert normalize_audio_path({"orig_name": str(wav)}) == str(wav)


def test_dict_with_missing_key() -> None:
    assert normalize_audio_path({}) is None


def test_dict_fallback_to_name_and_path(tmp_path: str) -> None:
    wav = tmp_path / "fallback.wav"
    sf.write(str(wav), np.zeros(44_100, dtype=np.float32), 44_100)

    assert normalize_audio_path({"name": str(wav)}) == str(wav)
    assert normalize_audio_path({"path": str(wav)}) == str(wav)


def test_dict_with_empty_keys_raises() -> None:
    assert normalize_audio_path({"orig_name": "", "name": None, "path": ""}) is None
