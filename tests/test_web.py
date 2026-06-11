"""Tests for web.server — _normalize_audio_path and serve."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from kuchikae.web import _normalize_audio_path


def _write_wav(path: str) -> None:
    data = np.zeros(16000, dtype=np.float32)
    sf.write(path, data, 16000)


def test_normalize_audio_path_none_raises() -> None:
    with pytest.raises(ValueError, match="Audio required"):
        _normalize_audio_path(None)


def test_normalize_audio_path_str_valid(tmp_path: str) -> None:
    wav = tmp_path / "test.wav"
    _write_wav(str(wav))
    assert _normalize_audio_path(str(wav)) == str(wav)


def test_normalize_audio_path_str_valid_mp3(tmp_path: str) -> None:
    wav = tmp_path / "test.mp3"
    _write_wav(str(wav))
    result = _normalize_audio_path(str(wav))
    assert result == str(wav)


def test_normalize_audio_path_str_nonexistent_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported audio"):
        _normalize_audio_path("/nonexistent/path.wav")


def test_normalize_audio_path_dict_with_path(tmp_path: str) -> None:
    wav = tmp_path / "dict.wav"
    _write_wav(str(wav))
    assert _normalize_audio_path({"path": str(wav)}) == str(wav)


def test_normalize_audio_path_dict_with_name(tmp_path: str) -> None:
    wav = tmp_path / "dict_name.wav"
    _write_wav(str(wav))
    assert _normalize_audio_path({"name": str(wav)}) == str(wav)


def test_normalize_audio_path_dict_with_orig_name(tmp_path: str) -> None:
    wav = tmp_path / "orig.wav"
    _write_wav(str(wav))
    assert _normalize_audio_path({"orig_name": str(wav)}) == str(wav)


def test_normalize_audio_path_dict_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="Audio required"):
        _normalize_audio_path({})


def test_normalize_audio_path_empty_dict_raises() -> None:
    with pytest.raises(ValueError, match="Audio required"):
        _normalize_audio_path({"path": None, "name": None})


def test_normalize_audio_path_too_long_raises(tmp_path: str) -> None:
    wav = tmp_path / "long.wav"
    _write_wav(str(wav))
    with pytest.raises(ValueError, match="Unsupported audio"):
        with patch("kuchikae.pipeline.audio_validation.sf.info") as mock_info:
            mock_info.return_value.duration = 30.0
            _normalize_audio_path(str(wav))


def test_normalize_audio_path_unsupported_extension_raises(tmp_path: str) -> None:
    txt = tmp_path / "test.txt"
    txt.write_text("not audio")
    with pytest.raises(ValueError, match="Unsupported audio"):
        _normalize_audio_path(str(txt))
