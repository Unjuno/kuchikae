"""Kuchikae web server entry point."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from kuchikae.domain.types import TextTransformPrompt, VoiceOutputPrompt
from kuchikae.pipeline.audio_validation import validate_audio
from kuchikae.pipeline import create_pipeline
from kuchikae.ui import CSS, create_app


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
    
    if os.path.splitext(path)[1].lower() not in {'.wav', '.mp3', '.m4a', '.flac'}:
        raise ValueError("Unsupported audio.")
    
    try:
        validate_audio(path)
    except ValueError as e:
        raise ValueError(str(e))
    
    return path


def serve() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG,
        format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    default_prompt = TextTransformPrompt.from_file()
    default_voice_prompt = VoiceOutputPrompt.from_file()
    
    # Check for streaming STT config via env var
    import os
    streaming_stt = os.environ.get("KUCHIKAE_STREAMING_STT", "").lower() in ("1", "true", "yes")
    pipeline = create_pipeline(
        {
            "stt_backend": os.environ.get("KUCHIKAE_STT_BACKEND", "faster_whisper"),
            "text_transform_backend": os.environ.get("KUCHIKAE_TEXT_BACKEND", "ollama"),
            "text_transform_model": os.environ.get("KUCHIKAE_TEXT_MODEL"),
            "voice_output_backend": os.environ.get("KUCHIKAE_VOICE_BACKEND", "irodori"),
            "streaming_stt": streaming_stt,
            "segmented_stt": os.environ.get("KUCHIKAE_SEGMENTED_STT", "").lower() in ("1", "true", "yes"),
            "allow_dummy_backends": os.environ.get("KUCHIKAE_ALLOW_DUMMY_BACKENDS", "").lower() in ("1", "true", "yes"),
        }
    )
    pipeline.warmup()
    demo = create_app(
        pipeline,
        default_prompt,
        default_voice_prompt,
        live_streaming=streaming_stt,
    )
    demo.launch(css=CSS, server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))


if __name__ == "__main__":
    serve()
