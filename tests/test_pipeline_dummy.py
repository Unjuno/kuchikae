"""Tests for KuchikaePipeline with dummy backends."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.domain.types import TextTransformPrompt


def _write_dummy_wav(path: str) -> None:
    samples = np.zeros(44_100, dtype=np.float32)
    sf.write(path, samples, 44_100)


def _run_pipeline(audio_path: str):
    pipeline = KuchikaePipeline()
    return pipeline.process(
        audio_path,
        TextTransformPrompt(instruction="確認"),
    )


def test_full_pipeline_runs(tmp_path):
    audio_path = str(tmp_path / "input.wav")
    _write_dummy_wav(audio_path)
    result = _run_pipeline(audio_path)
    assert os.path.isfile(result.output_audio_path)


def test_pipeline_output_audio_exists(tmp_path):
    audio_path = str(tmp_path / "input.wav")
    _write_dummy_wav(audio_path)
    result = _run_pipeline(audio_path)
    assert os.path.isfile(result.output_audio_path)
