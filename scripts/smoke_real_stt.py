#!/usr/bin/env python3
"""Smoke test for real STT (faster-whisper).

Requires: faster-whisper installed.

Usage:
    uv run python scripts/smoke_real_stt.py
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import soundfile as sf

from kuchikae.domain.stt import DummySTTBackend
from kuchikae.backends.stt import FasterWhisperSTTBackend


def create_test_audio(path: str) -> None:
    """Create a short test WAV file."""
    samples = np.zeros(44_100).astype(np.float32)  # silent for now.
    sf.write(path, samples, 44_100)


def main() -> int:
    print("=== STT Smoke Test ===")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name
    create_test_audio(audio_path)
    print(f"  Audio: {audio_path}")

    # Dummy.
    dummy = DummySTTBackend()
    result1 = dummy.transcribe(audio_path)
    print(f"\n--- DummySTTBackend ---")
    print(f"  Result: {result1}")

    # Faster-whisper (if available).
    try:
        fw = FasterWhisperSTTBackend()
        result2 = fw.transcribe(audio_path)
        print(f"\n--- FasterWhisperSTTBackend ---")
        print(f"  Result: {result2}")
    except RuntimeError as e:
        print(f"\n--- FasterWhisperSTTBackend SKIPPED ---")
        print(f"  Reason: {e}")

    os.unlink(audio_path)
    print("\n=== PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
