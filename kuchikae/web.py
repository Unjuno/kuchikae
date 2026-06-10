"""Kuchikae web server entry point."""

from __future__ import annotations

import logging
import sys


def serve() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG,
        format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from kuchikae.pipeline import create_pipeline
    from kuchikae.types import TextTransformPrompt
    from kuchikae.ui import CSS, create_app

    default_prompt = TextTransformPrompt.from_file()
    pipeline = create_pipeline()
    pipeline.warmup()
    demo = create_app(pipeline, default_prompt)
    demo.launch(css=CSS)


if __name__ == "__main__":
    serve()
