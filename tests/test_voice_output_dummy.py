"""Tests for DummyVoiceOutputBackend."""

import os
import tempfile

import numpy as np
import soundfile as sf

from kuchikae.types import VoiceContext, VoiceOutputPrompt
from kuchikae.voice_output import DummyVoiceOutputBackend


def test_dummy_voice_output_creates_wav():
    backend = DummyVoiceOutputBackend()
    voice_ctx = VoiceContext(voice_id="test")
    prompt = VoiceOutputPrompt(prompt_text="自然に")

    path = backend.synthesize("テスト", voice_ctx, prompt)
    assert os.path.isfile(path)


def test_dummy_wav_can_be_read_by_soundfile():
    backend = DummyVoiceOutputBackend()
    voice_ctx = VoiceContext(voice_id="test")
    prompt = VoiceOutputPrompt(prompt_text="自然に")

    path = backend.synthesize("読み込みテスト", voice_ctx, prompt)

    data, sr = sf.read(path)
    assert isinstance(data, np.ndarray)
    assert sr == 44100


def test_wav_file_is_valid():
    """Verify the WAV file is non-empty and has content."""
    backend = DummyVoiceOutputBackend()
    voice_ctx = VoiceContext(voice_id="test")
    prompt = VoiceOutputPrompt(prompt_text="確認")

    path = backend.synthesize("ファイル検証", voice_ctx, prompt)
    assert os.path.getsize(path) > 0
