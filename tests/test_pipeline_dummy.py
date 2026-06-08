"""Tests for KuchikaePipeline with dummy backends."""

import os
import tempfile

import numpy as np
import soundfile as sf

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.types import TextTransformPrompt, VoiceOutputPrompt


def _write_dummy_wav(path: str) -> None:
    samples = np.zeros(44_100, dtype=np.float32)  # 1 sec at 44.1 kHz
    sf.write(path, samples, 44100)


def test_full_pipeline_runs():
    pipeline = KuchikaePipeline()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)
        audio_path = f.name

    text_prompt = TextTransformPrompt(prompt_text="丁寧にして")
    voice_prompt = VoiceOutputPrompt(prompt_text="自然に")

    result = pipeline.process(audio_path, text_prompt, voice_prompt)

    assert os.path.isfile(result.output_audio_path)


def test_pipeline_source_text_exists():
    pipeline = KuchikaePipeline()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)
        audio_path = f.name

    result = pipeline.process(
        audio_path,
        TextTransformPrompt(prompt_text="確認"),
        VoiceOutputPrompt(prompt_text="確認"),
    )

    assert len(result.source_text) > 0


def test_pipeline_transformed_text_exists():
    pipeline = KuchikaePipeline()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)
        audio_path = f.name

    result = pipeline.process(
        audio_path,
        TextTransformPrompt(prompt_text="確認"),
        VoiceOutputPrompt(prompt_text="確認"),
    )

    assert len(result.transformed_text) > 0


def test_pipeline_output_audio_exists():
    pipeline = KuchikaePipeline()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)
        audio_path = f.name

    result = pipeline.process(
        audio_path,
        TextTransformPrompt(prompt_text="確認"),
        VoiceOutputPrompt(prompt_text="確認"),
    )

    assert os.path.isfile(result.output_audio_path)


def test_pipeline_voice_ready():
    pipeline = KuchikaePipeline()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)
        audio_path = f.name

    result = pipeline.process(
        audio_path,
        TextTransformPrompt(prompt_text="確認"),
        VoiceOutputPrompt(prompt_text="確認"),
    )

    assert result.voice_context.ready is True


def test_pipeline_latency_fields_non_negative():
    pipeline = KuchikaePipeline()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)
        audio_path = f.name

    result = pipeline.process(
        audio_path,
        TextTransformPrompt(prompt_text="確認"),
        VoiceOutputPrompt(prompt_text="確認"),
    )

    assert result.latency.stt_seconds >= 0
    assert result.latency.text_transform_seconds >= 0
    assert result.latency.voice_output_seconds >= 0
    assert result.latency.total_seconds >= 0
