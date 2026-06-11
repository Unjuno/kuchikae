"""Kuchikae v0.1 — entry point."""

from __future__ import annotations

import logging
import os
import sys

from kuchikae.domain.types import TextTransformPrompt, VoiceOutputPrompt
from kuchikae.pipeline import create_pipeline
from kuchikae.ui import CSS, create_app


def main() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG,
        format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    default_prompt = TextTransformPrompt.from_file()
    default_voice_prompt = VoiceOutputPrompt.from_file()
    pipeline = create_pipeline(
        {
            "stt_backend": os.environ.get("KUCHIKAE_STT_BACKEND", "faster_whisper"),
            "text_transform_backend": os.environ.get("KUCHIKAE_TEXT_BACKEND", "ollama"),
            "text_transform_model": os.environ.get("KUCHIKAE_TEXT_MODEL"),
            "voice_output_backend": os.environ.get("KUCHIKAE_VOICE_BACKEND", "irodori"),
            "streaming_stt": os.environ.get("KUCHIKAE_STREAMING_STT", "").lower() in ("1", "true", "yes"),
            "segmented_stt": os.environ.get("KUCHIKAE_SEGMENTED_STT", "").lower() in ("1", "true", "yes"),
            "allow_dummy_backends": os.environ.get("KUCHIKAE_ALLOW_DUMMY_BACKENDS", "").lower() in ("1", "true", "yes"),
        }
    )
    pipeline.warmup()
    demo = create_app(pipeline, default_prompt, default_voice_prompt)
    demo.launch(css=CSS, server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))


if __name__ == "__main__":
    main()
