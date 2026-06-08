"""Tests for DummyVoiceOutputBackend."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from kuchikae.types import VoiceContext, VoiceOutputPrompt
from kuchikae.voice_output import DummyVoiceOutputBackend


def _voice_context() -> VoiceContext:
    return VoiceContext(reference_audio_path="tests/fixtures/reference.wav", ready=True)


def test_dummy_voice_output_creates_wav():
    backend = DummyVoiceOutputBackend()
    voice_ctx = _voice_context()
    prompt = VoiceOutputPrompt(instruction="自然に")

    path = backend.synthesize("テスト", voice_ctx, prompt)

    assert os.path.isfile(path)


def test_dummy_wav_can_be_read_by_soundfile():
    backend = DummyVoiceOutputBackend()
    voice_ctx = _voice_context()
    prompt = VoiceOutputPrompt(instruction="自然に")

    path = backend.synthesize("読み込みテスト", voice_ctx, prompt)

    data, sr = sf.read(path)
    assert isinstance(data, np.ndarray)
    assert sr == 44_100


def test_wav_file_is_valid():
    backend = DummyVoiceOutputBackend()
    voice_ctx = _voice_context()
    prompt = VoiceOutputPrompt(instruction="確認")

    path = backend.synthesize("ファイル検証", voice_ctx, prompt)

    assert os.path.getsize(path) > 0
