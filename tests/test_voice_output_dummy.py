"""Tests for DummyVoiceOutputBackend."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from kuchikae.domain.voice_output import DummyVoiceOutputBackend


def _write_dummy_wav(path: str) -> None:
    samples = np.zeros(44_100, dtype=np.float32)
    sf.write(path, samples, 44_100)


def test_dummy_voice_output_creates_wav(tmp_path):
    backend = DummyVoiceOutputBackend()
    audio_path = str(tmp_path / "ref.wav")
    _write_dummy_wav(audio_path)
    path = backend.synthesize("テスト", audio_path)
    assert os.path.isfile(path)


def test_dummy_wav_can_be_read_by_soundfile(tmp_path):
    backend = DummyVoiceOutputBackend()
    audio_path = str(tmp_path / "ref.wav")
    _write_dummy_wav(audio_path)
    path = backend.synthesize("読み込みテスト", audio_path)
    data, sr = sf.read(path)
    assert isinstance(data, np.ndarray)
    assert sr == 44_100


def test_wav_file_is_valid(tmp_path):
    backend = DummyVoiceOutputBackend()
    audio_path = str(tmp_path / "ref.wav")
    _write_dummy_wav(audio_path)
    path = backend.synthesize("ファイル検証", audio_path)
    assert os.path.getsize(path) > 0
