"""Tests for VoiceContextExtractor."""

from __future__ import annotations

import tempfile

import numpy as np
import soundfile as sf

from kuchikae.types import VoiceContext
from kuchikae.voice_context import VoiceContextExtractor


def _write_dummy_wav(path: str) -> None:
    samples = np.zeros(44_100, dtype=np.float32)
    sf.write(path, samples, 44_100)


def test_extractor_returns_not_ready_context_without_reference():
    extractor = VoiceContextExtractor()

    ctx = extractor.extract(None)

    assert isinstance(ctx, VoiceContext)
    assert ctx.ready is False
    assert ctx.reference_audio_path == ""


def test_extractor_returns_ready_context_with_reference_path():
    extractor = VoiceContextExtractor()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)
        ref = f.name

    ctx = extractor.extract(ref)

    assert isinstance(ctx, VoiceContext)
    assert ctx.ready is True
    assert ctx.reference_audio_path == ref
