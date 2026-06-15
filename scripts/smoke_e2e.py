#!/usr/bin/env python3
"""End-to-end smoke test with real backends.

Requires:
- OpenVoice repo at $KUCHIKAE_OPENVOICE_PATH or ../OpenVoice.
- Checkpoint files downloaded from HuggingFace (see docs/REAL_MODEL_SETUP.md).
- torch, torchaudio installed.

Usage:
    OPENVOICE_READY=1 uv run python scripts/smoke_e2e.py
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import soundfile as sf

from kuchikae.pipeline import create_pipeline
from kuchikae.domain.types import TextTransformPrompt


def main() -> int:
    print("=== E2E Smoke Test (real backends) ===")

    # Create test audio.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name
    try:
        samples = np.random.randn(4 * 44_100).astype(np.float32)
        sf.write(audio_path, samples, 44_100)

        # Create pipeline with real backends.
        os.environ.setdefault("OPENVOICE_READY", "1")
        try:
            pipe = create_pipeline({"text_transform_backend": "rule"})
        except RuntimeError as e:
            print(f"SKIPPED (backends not ready): {e}")
            return 1

        # Run pipeline.
        result = pipe.process(
            audio_path=audio_path,
            text_transform_prompt=TextTransformPrompt(instruction="確認"),
        )

        print(f"\n--- PipelineResult ---")
        print(f"  Output audio: {result.output_audio_path} ({os.path.getsize(result.output_audio_path)} bytes)")

        # Verify output.
        assert os.path.isfile(result.output_audio_path), "Output audio missing!"
        data, sr = sf.read(result.output_audio_path)
        print(f"\n  Output duration: {len(data)/sr:.2f}s at {sr} Hz")

        print("\n=== PASS ===")
        return 0
    finally:
        os.unlink(audio_path)


if __name__ == "__main__":
    raise SystemExit(main())
