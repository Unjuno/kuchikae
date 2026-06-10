"""Tests for app.py — audio input normalization."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
from gradio import Error as GrError

from app import _normalize_audio_path


def test_str_existing_file(tmp_path: str) -> None:
    wav = tmp_path / "x.wav"
    sf.write(str(wav), np.zeros(44_100, dtype=np.float32), 44_100)
    assert _normalize_audio_path(str(wav)) == str(wav)


def test_str_nonexistent_file(tmp_path: str) -> None:
    with pytest.raises(GrError, match="No audio file found"):
        _normalize_audio_path(str(tmp_path / "missing.wav"))


def test_none() -> None:
    with pytest.raises(GrError, match="No audio input received"):
        _normalize_audio_path(None)


def test_tuple_returns_tmp_path() -> None:
    sr = 48_000
    data = np.zeros(sr, dtype=np.float32)
    path = _normalize_audio_path((sr, data))
    assert isinstance(path, str) and os.path.isfile(path)


def test_dict_with_orig_name(tmp_path: str) -> None:
    wav = tmp_path / "orig.wav"
    sf.write(str(wav), np.zeros(44_100, dtype=np.float32), 44_100)
    assert _normalize_audio_path({"orig_name": str(wav)}) == str(wav)


def test_dict_with_missing_key() -> None:
    with pytest.raises(GrError):
        _normalize_audio_path({})
