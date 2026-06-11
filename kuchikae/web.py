"""Kuchikae web server entry point."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from typing import Any

import numpy as np
import soundfile as sf

from kuchikae.pipeline import create_pipeline
from kuchikae.types import TextTransformPrompt
from kuchikae.ui import CSS, create_app

ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac'}
MAX_FILE_SIZE = 25 * 1024 * 1024
MAX_AUDIO_DURATION = 20.0


def _normalize_audio_path(audio_input: Any) -> str:
    """Normalize audio input to file path with validation.
    
    Supports:
    - str: file path
    - dict with path or name
    - object with .path
    
    Rejects missing or invalid files with short errors.
    """
    if audio_input is None:
        raise ValueError("Audio required.")
    
    # Extract path from various input types
    if isinstance(audio_input, str):
        path = audio_input
    elif isinstance(audio_input, dict):
        path = (
            audio_input.get("path")
            or audio_input.get("name")
            or audio_input.get("orig_name")
        )
    else:
        path = (
            getattr(audio_input, "path", None)
            or getattr(audio_input, "name", None)
            or getattr(audio_input, "orig_name", None)
        )
    
    if not path:
        raise ValueError("Audio required.")
    
    path = str(path)
    
    # Check if file exists
    if not os.path.isfile(path):
        raise ValueError("Unsupported audio.")
    
    # Validate extension
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported audio.")
    
    # Validate file size
    try:
        size = os.path.getsize(path)
        if size > MAX_FILE_SIZE:
            raise ValueError("File too large.")
    except OSError:
        raise ValueError("Unsupported audio.")
    
    # Validate duration
    try:
        info = sf.info(path)
        if info.duration > MAX_AUDIO_DURATION:
            raise ValueError("Max 20s.")
    except Exception:
        raise ValueError("Unsupported audio.")
    
    return path


def serve() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG,
        format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    default_prompt = TextTransformPrompt.from_file()
    
    # Check for streaming STT config via env var
    import os
    streaming_stt = os.environ.get("KUCHIKAE_STREAMING_STT", "").lower() in ("1", "true", "yes")
    
    pipeline = create_pipeline({"streaming_stt": streaming_stt} if streaming_stt else None)
    pipeline.warmup()
    demo = create_app(pipeline, default_prompt, live_streaming=streaming_stt)
    demo.launch(css=CSS)


if __name__ == "__main__":
    serve()
