"""Kuchikae CLI entry point."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print("Usage: kuchikae")
        print("Starts the Kuchikae web server at http://127.0.0.1:7860")
        return

    from kuchikae.web import serve

    serve()
