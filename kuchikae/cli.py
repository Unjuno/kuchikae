"""Kuchikae CLI entry point."""

from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        port = os.environ.get("GRADIO_SERVER_PORT", "7860")
        print("Usage: kuchikae [--streaming]")
        print("  --streaming  Enable streaming STT for push-to-talk (partial transcripts)")
        print(f"Starts the Kuchikae web server at http://127.0.0.1:{port}")
        return

    streaming = "--streaming" in sys.argv
    if streaming:
        os.environ["KUCHIKAE_STREAMING_STT"] = "1"

    from kuchikae.web import serve

    serve()
