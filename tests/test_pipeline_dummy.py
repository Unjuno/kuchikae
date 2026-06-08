"""Tests for KuchikaePipeline with dummy backends."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import soundfile as sf

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.types import TextTransformPrompt, VoiceOutputPrompt


def _write_dummy_wav(path: str) -> None:
    samples = np.zeros(44_100, dtype=np.float32)
    sf.write(path, samples, 44_100)


def _run_pipeline():
    pipeline = KuchikaePipeline()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_dummy_wav(f.name)
        audio_path = f.name

    return pipeline.process(
        audio_path,
        TextTransformPrompt(instruction="確認"),
        VoiceOutputPrompt(instruction="自然に"),
    )


def test_full_pipeline_runs():
    result = _run_pipeline()

    assert os.path.isfile(result.output_audio_path)


def test_pipeline_source_text_exists():
    result = _run_pipeline()

    assert result.source_text


def test_pipeline_transformed_text_exists():
    result = _run_pipeline()

    assert result.transformed_text


def test_pipeline_output_audio_exists():
    result = _run_pipeline()

    assert os.path.isfile(result.output_audio_path)


def test_pipeline_voice_ready():
    result = _run_pipeline()

    assert result.voice_ready is True


def test_pipeline_prompt_fields_are_preserved():
    result = _run_pipeline()

    assert result.text_transform_prompt == "確認"
    assert result.voice_output_prompt == "自然に"


def test_pipeline_latency_fields_non_negative():
    result = _run_pipeline()

    assert result.latency.stt_seconds >= 0
    assert result.latency.text_transform_seconds >= 0
    assert result.latency.voice_output_seconds >= 0
    assert result.latency.total_seconds >= 0
