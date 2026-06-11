"""Shared audio validation constants and helpers.

These are the canonical values used by both the pipeline (internal processing)
and the web entry point (user-facing validation).
"""

from __future__ import annotations

import os

import soundfile as sf

# Supported audio formats (user-facing: allow common formats)
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac'}
MAX_FILE_SIZE = 25 * 1024 * 1024
MAX_AUDIO_DURATION = 20.0


def validate_audio(audio_path: str) -> None:
    """Validate an audio file's extension, size, and duration.

    Raises ValueError with a descriptive message on failure.
    """
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported format ({ext})")
    size = os.path.getsize(audio_path)
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large ({size / 1e6:.1f}MB > {MAX_FILE_SIZE / 1e6:.0f}MB)")
    info = sf.info(audio_path)
    if info.duration > MAX_AUDIO_DURATION:
        raise ValueError(f"Audio too long ({info.duration:.0f}s > {MAX_AUDIO_DURATION:.0f}s)")
