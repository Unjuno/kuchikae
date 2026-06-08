"""Tests for VoiceContextExtractor."""

import tempfile

from kuchikae.audio_cache import AudioCache
from kuchikae.voice_context import VoiceContextExtractor


def _write_dummy_wav(path: str) -> None:
    """Write a short silent WAV file using numpy + soundfile."""
    import numpy as np
    import soundfile as sf

    samples = np.zeros(44_100, dtype=np.float32)
    sf.write(path, samples, 44100)


def test_extractor_returns_voice_context():
    extractor = VoiceContextExtractor()
    cache = AudioCache()
    ctx = extractor.extract(cache)

    assert isinstance(ctx, type(extractor.extract(cache)))
    assert ctx.voice_id != ""


def test_ready_false_without_reference():
    extractor = VoiceContextExtractor()
    cache = AudioCache()
    # Don't set a reference path
    ctx = extractor.extract(cache)
    assert ctx.ready is False


def test_ready_true_with_reference():
    extractor = VoiceContextExtractor()
    cache = AudioCache()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)
        ref = f.name

    cache.set_reference_audio(ref)
    ctx = extractor.extract(cache)
    assert ctx.ready is True
    assert ctx.reference_path == ref
