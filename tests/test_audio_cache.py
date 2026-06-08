"""Tests for AudioCache."""

import os
import tempfile

import pytest

from kuchikae.audio_cache import AudioCache


def _write_dummy_wav(path: str) -> None:
    """Write a short silent WAV file using numpy + soundfile."""
    import numpy as np
    import soundfile as sf

    samples = np.zeros(44_100, dtype=np.float32)  # 1 sec at 44.1 kHz
    sf.write(path, samples, 44100)


def test_add_utterance():
    cache = AudioCache()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)
        path = f.name

    cache.add_utterance(path)
    assert len(cache.utterances) == 1
    assert cache.latest_utterance_path == path


def test_latest_utterance_updated():
    cache = AudioCache()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)

    p1 = f.name
    p2 = os.path.join(tempfile.gettempdir(), "another.wav")
    with open(p2, "w"):
        pass  # touch it

    cache.add_utterance(p1)
    assert cache.latest_utterance_path == p1

    cache.add_utterance(p2)
    assert cache.latest_utterance_path == p2


def test_reference_audio_path():
    cache = AudioCache()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)
        ref = f.name

    assert cache.reference_path is None
    cache.set_reference_audio(ref)
    assert cache.reference_path == ref


def test_empty_cache():
    cache = AudioCache()
    assert cache.utterances == []
    assert cache.latest_utterance_path is None
    assert cache.reference_path is None
