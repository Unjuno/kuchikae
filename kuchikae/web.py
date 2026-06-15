"""Kuchikae web server entry point."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from kuchikae.domain.types import TextTransformPrompt, VoiceOutputPrompt
from kuchikae.pipeline.audio_validation import validate_audio
from kuchikae.pipeline import create_pipeline
from kuchikae.ui import CSS, THEME, create_app


def _truthy(value: str | None) -> bool:
    return (value or "").lower() in ("1", "true", "yes", "on")


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
        path: str = audio_input
    elif isinstance(audio_input, dict):
        path = str(
            audio_input.get("path")
            or audio_input.get("name")
            or audio_input.get("orig_name")
            or ""
        )
    else:
        path = str(
            getattr(audio_input, "path", None)
            or getattr(audio_input, "name", None)
            or getattr(audio_input, "orig_name", None)
            or ""
        )
    
    if not path:
        raise ValueError("Audio required.")
    
    if os.path.splitext(path)[1].lower() not in {'.wav', '.mp3', '.m4a', '.flac'}:
        raise ValueError("Unsupported audio.")
    
    try:
        validate_audio(path)
    except ValueError as e:
        raise ValueError(str(e)) from e
    
    return path


def build_pipeline_config_from_env(
    dummy: bool = False,
    real: bool = False,
    streaming: bool | None = None,
) -> dict:
    """Build pipeline config from environment variables.
    
    Args:
        dummy: Use dummy backends for smoke testing
        real: Use real backends (requires models)
        streaming: Enable streaming STT (None = check env var)
    """
    requested_streaming = (
        bool(streaming)
        if streaming is not None
        else _truthy(os.environ.get("KUCHIKAE_STREAMING_STT"))
    )
    streaming_stt = requested_streaming and real
    if requested_streaming and not real:
        logging.getLogger("kuchikae.web").warning(
            "streaming STT requires --real; disabling streaming for smoke-friendly mode"
        )
    
    common = {
        "stt_preset": os.environ.get("KUCHIKAE_STT_PRESET", "balanced"),
        "text_transform_model": os.environ.get("KUCHIKAE_TEXT_MODEL"),
        "streaming_stt": streaming_stt,
        "segmented_stt": _truthy(os.environ.get("KUCHIKAE_SEGMENTED_STT")),
    }
    
    if dummy:
        return {
            **common,
            "stt_backend": "dummy",
            "text_transform_backend": "prompted_rule",
            "voice_output_backend": "dummy",
            "allow_dummy_backends": True,
        }
    
    if real:
        return {
            **common,
            "stt_backend": os.environ.get("KUCHIKAE_STT_BACKEND", "faster_whisper"),
            "text_transform_backend": os.environ.get("KUCHIKAE_TEXT_BACKEND", "ollama"),
            "voice_output_backend": os.environ.get("KUCHIKAE_VOICE_BACKEND", "irodori"),
            "allow_dummy_backends": False,
        }
    
    return {
        **common,
        "stt_backend": os.environ.get("KUCHIKAE_STT_BACKEND", "dummy"),
        "text_transform_backend": os.environ.get("KUCHIKAE_TEXT_BACKEND", "prompted_rule"),
        "voice_output_backend": os.environ.get("KUCHIKAE_VOICE_BACKEND", "dummy"),
        "allow_dummy_backends": _truthy(os.environ.get("KUCHIKAE_ALLOW_DUMMY_BACKENDS", "1")),
    }


def serve(
    dummy: bool = False,
    real: bool = False,
    streaming: bool | None = None,
    port: int | None = None,
) -> None:
    """Start the Kuchikae web server.
    
    Args:
        dummy: Use dummy backends for smoke testing
        real: Use real backends (requires models)
        streaming: Enable streaming STT
        port: Server port (default: 7860)
    """
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG,
        format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    default_prompt = TextTransformPrompt.from_file()
    default_voice_prompt = VoiceOutputPrompt.from_file()
    
    config = build_pipeline_config_from_env(dummy=dummy, real=real, streaming=streaming)
    pipeline = create_pipeline(config)
    pipeline.warmup()
    
    live_streaming = config.get("streaming_stt", False)
    demo = create_app(
        pipeline,
        default_prompt,
        default_voice_prompt,
        live_streaming=live_streaming,
    )
    
    server_port = port or int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(css=CSS, server_port=server_port, theme=THEME)


if __name__ == "__main__":
    serve()
