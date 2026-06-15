#!/usr/bin/env python3
"""Generate WAV fixtures for voice eval using macOS say command.

Usage:
  uv run python evals/create_fixtures.py
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "voice"

FIXTURES: dict[str, str] = {
    "neutral_short.wav": "明日の会議は14時からです。",
    "neutral_long.wav": "来週の月曜日に3000円の交通費を申請してください。大阪駅で14時30分に待ち合わせです。",
    "happy_greeting.wav": "こんにちは！今日は本当にいい天気ですね。",
    "calm_instruction.wav": "ゆっくりで大丈夫ですよ。間違えても気にしないでください。",
    "short_yes.wav": "はい。",
    "short_no.wav": "いいえ、違います。",
}


def wav_exists(name: str) -> bool:
    return (FIXTURES_DIR / name).exists()


def generate_fixture(name: str, text: str) -> None:
    if platform.system() != "Darwin":
        logger.error("create_fixtures requires macOS (say command)")
        sys.exit(1)
    out = FIXTURES_DIR / name
    logger.info("generating %s …", name)
    result = subprocess.run(
        ["say", "-o", str(out), "--data-format=LEI16@16000", text],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("say failed for %s: %s", name, result.stderr)
        sys.exit(1)
    size = out.stat().st_size
    logger.info("  done: %s (%d bytes)", name, size)


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for n in FIXTURES_DIR.iterdir() if n.suffix == ".wav")
    logger.info("fixtures dir: %s (%d existing WAVs)", FIXTURES_DIR, existing)

    for name, text in FIXTURES.items():
        if wav_exists(name) and (FIXTURES_DIR / name).stat().st_size > 0:
            logger.info("skipping %s (already exists)", name)
            continue
        generate_fixture(name, text)


if __name__ == "__main__":
    main()
