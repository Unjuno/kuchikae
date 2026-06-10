"""Kuchikae v0.1 — entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from kuchikae.pipeline import create_pipeline
from kuchikae.types import TextTransformPrompt
from kuchikae.ui import CSS, create_app


def main() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG,
        format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    default_prompt = TextTransformPrompt.from_file(Path("prompts/text_transform_default.txt"))
    pipeline = create_pipeline()
    pipeline.warmup()
    demo = create_app(pipeline, default_prompt)
    demo.launch(css=CSS)


if __name__ == "__main__":
    main()
