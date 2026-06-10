"""Tests for KuchikaePipeline with dummy backends."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import soundfile as sf

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.types import TextTransformPrompt


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
    )


def test_full_pipeline_runs():
    result = _run_pipeline()

    assert os.path.isfile(result.output_audio_path)


def test_pipeline_output_audio_exists():
    result = _run_pipeline()

    assert os.path.isfile(result.output_audio_path)
