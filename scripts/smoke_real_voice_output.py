#!/usr/bin/env python3
"""Smoke test for real voice output (OpenVoice).

Requires:
- OpenVoice repo at $KUCHIKAE_OPENVOICE_PATH or ../OpenVoice.
- Checkpoint files downloaded from HuggingFace.
  See docs/REAL_MODEL_SETUP.md for how to download them.
- torch, torchaudio installed.

Usage:
    uv run python scripts/smoke_real_voice_output.py
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import soundfile as sf

from kuchikae.domain.voice_output import DummyVoiceOutputBackend
from kuchikae.backends.voice_output import OpenVoiceOutputBackend
from kuchikae.domain.types import VoiceContext


def create_test_audio(path: str) -> None:
    """Create a short test WAV file."""
    samples = np.random.randn(2 * 44_100).astype(np.float32)
    sf.write(path, samples, 44_100)


def main() -> int:
    print("=== OpenVoice Voice Output Smoke Test ===")

    # Create test audio.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ref_audio = f.name
    dummy_path = None
    ow_path = None
    try:
        create_test_audio(ref_audio)
        print(f"  Reference audio: {ref_audio}")

        # Test dummy backend.
        print("\n--- DummyVoiceOutputBackend ---")
        dummy_backend = DummyVoiceOutputBackend()
        voice_context_dummy = VoiceContext(reference_audio_path=ref_audio, ready=True)
        dummy_path = dummy_backend.synthesize("テスト", voice_context_dummy)
        assert os.path.isfile(dummy_path), f"Dummy output missing: {dummy_path}"
        data, sr = sf.read(dummy_path)
        print(f"  Output: {dummy_path}")
        print(f"  Duration: {len(data)/sr:.2f}s, SR: {sr}")

        # Test OpenVoice backend.
        print("\n--- OpenVoiceOutputBackend ---")
        try:
            ow_backend = OpenVoiceOutputBackend()
            voice_context = VoiceContext(reference_audio_path=ref_audio, ready=True)
            ow_path = ow_backend.synthesize("テスト", voice_context)
            assert os.path.isfile(ow_path), f"OpenVoice output missing: {ow_path}"
            data2, sr2 = sf.read(ow_path)
            print(f"  Output: {ow_path}")
            print(f"  Duration: {len(data2)/sr2:.2f}s, SR: {sr2}")
        except RuntimeError as e:
            print(f"  SKIPPED (OpenVoice not ready): {e}")
            return 1

        print("\n=== PASS ===")
        return 0
    finally:
        os.unlink(ref_audio)
        for p in [dummy_path, ow_path]:
            if p and os.path.isfile(p):
                os.unlink(p)


if __name__ == "__main__":
    raise SystemExit(main())
