"""Tests for AudioCache."""

from __future__ import annotations

import tempfile

import numpy as np
import soundfile as sf

from kuchikae.audio_cache import AudioCache


def _write_dummy_wav(path: str) -> None:
    samples = np.zeros(44_100, dtype=np.float32)
    sf.write(path, samples, 44_100)


def test_add_utterance_sets_latest_and_reference_audio_path():
    cache = AudioCache()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)
        path = f.name

    cache.add_utterance(path)

    assert cache.get_latest_utterance_path() == path
    assert cache.get_reference_audio_path() == path


def test_empty_cache_returns_none_paths():
    cache = AudioCache()

    assert cache.get_latest_utterance_path() is None
    assert cache.get_reference_audio_path() is None
