#!/usr/bin/env python3
"""Run voice/audio evaluation cases.

Usage:
  uv run python evals/run_voice_eval.py --dry-run
  uv run python evals/run_voice_eval.py --backend irodori
  uv run python evals/run_voice_eval.py --dry-run --out evals/results/voice_eval_smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CASES_PATH = Path(__file__).parent / "voice_cases.yaml"
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "voice"
RESULTS_DIR = Path(__file__).parent / "results"

Verdict = Literal["pass", "warn", "fail", "skip"]


# ---------------------------------------------------------------------------
# Schema: eval case
# ---------------------------------------------------------------------------

@dataclass
class ExpectedVoice:
    emotion: str = "neutral"
    voice_style: str = "auto"
    should_preserve: list[str] = field(default_factory=list)
    should_not_preserve: list[str] = field(default_factory=list)


@dataclass
class VoiceEvalCase:
    id: str
    input_audio: str
    input_text: str
    template: str
    expected: ExpectedVoice = field(default_factory=ExpectedVoice)


# ---------------------------------------------------------------------------
# Schema: output JSONL row
# ---------------------------------------------------------------------------

@dataclass
class VoiceEvalResult:
    case_id: str
    input_audio: str
    output_audio: str | None
    template: str
    input_text: str
    transformed_text: str | None
    voice_backend: str
    speaker_similarity: float | None = None
    duration_ratio: float | None = None
    pitch_delta_mean: float | None = None
    energy_delta_db: float | None = None
    input_emotion: str | None = None
    output_emotion: str | None = None
    verdict: Verdict = "skip"
    failure_reason: str = ""


# ---------------------------------------------------------------------------
# Load cases
# ---------------------------------------------------------------------------

def load_cases() -> list[VoiceEvalCase]:
    """Load voice eval cases from YAML."""
    with open(CASES_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cases: list[VoiceEvalCase] = []
    for raw in data.get("cases", []):
        exp_raw = raw.get("expected", {})
        expected = ExpectedVoice(
            emotion=exp_raw.get("emotion", "neutral"),
            voice_style=exp_raw.get("voice_style", "auto"),
            should_preserve=exp_raw.get("should_preserve", []),
            should_not_preserve=exp_raw.get("should_not_preserve", []),
        )
        cases.append(VoiceEvalCase(
            id=raw["id"],
            input_audio=raw["input_audio"],
            input_text=raw["input_text"],
            template=raw.get("template", "自然に"),
            expected=expected,
        ))
    return cases


# ---------------------------------------------------------------------------
# Validate fixtures
# ---------------------------------------------------------------------------

def validate_fixtures(cases: list[VoiceEvalCase]) -> dict[str, str]:
    """Check that fixture files exist. Returns {case_id: error|''}."""
    errors: dict[str, str] = {}
    for case in cases:
        fixture_path = FIXTURES_DIR / case.input_audio
        if not fixture_path.exists():
            errors[case.id] = f"fixture not found: {fixture_path}"
        else:
            if fixture_path.stat().st_size == 0:
                errors[case.id] = f"fixture is empty: {fixture_path}"
    return errors


# ---------------------------------------------------------------------------
# Process a single case
# ---------------------------------------------------------------------------

def process_case(
    case: VoiceEvalCase,
    backend: str,
    dry_run: bool = False,
) -> VoiceEvalResult:
    """Process a single voice eval case.

    In dry-run mode, skips actual inference and returns a placeholder result.
    When real audio analysis dependencies are missing, gracefully skips
    acoustic metric computation.
    """
    fixture_path = FIXTURES_DIR / case.input_audio
    if not fixture_path.exists():
        return VoiceEvalResult(
            case_id=case.id,
            input_audio=case.input_audio,
            output_audio=None,
            template=case.template,
            input_text=case.input_text,
            transformed_text=None,
            voice_backend=backend,
            verdict="skip",
            failure_reason=f"fixture not found: {fixture_path}",
        )

    if dry_run:
        logger.info("[dry-run] case=%s template=%s fixture=%s", case.id, case.template, fixture_path)
        return VoiceEvalResult(
            case_id=case.id,
            input_audio=case.input_audio,
            output_audio=None,
            template=case.template,
            input_text=case.input_text,
            transformed_text=None,
            voice_backend=backend,
            verdict="skip",
            failure_reason="dry-run (no inference)",
        )

    # ── Real inference path (future) ──
    # 1. Run text transform via pipeline
    # 2. Run TTS via chosen backend
    # 3. Compare input/output audio:
    #    - speaker_similarity (e.g. speechbrain ECAPA)
    #    - duration_ratio = output_duration / input_duration
    #    - pitch_delta_mean (e.g. pyworld or librosa)
    #    - energy_delta_db (RMS-based)
    #    - input_emotion (audio emotion detector on input)
    #    - output_emotion (audio emotion detector on output)
    # 4. Determine verdict:
    #    - skip: fixture missing or deps unavailable
    #    - pass: metrics within expected thresholds
    #    - warn: minor degradation
    #    - fail: major quality loss or error
    #
    # For now, return skip with a descriptive reason.

    try:
        # Attempt to import heavy dependencies — graceful skip if missing
        import librosa  # noqa: F401
        import numpy as np  # noqa: F401
        _ = fixture_path
    except ImportError:
        return VoiceEvalResult(
            case_id=case.id,
            input_audio=case.input_audio,
            output_audio=None,
            template=case.template,
            input_text=case.input_text,
            transformed_text=None,
            voice_backend=backend,
            verdict="skip",
            failure_reason="audio analysis dependencies not available (librosa, numpy)",
        )

    return VoiceEvalResult(
        case_id=case.id,
        input_audio=case.input_audio,
        output_audio=None,
        template=case.template,
        input_text=case.input_text,
        transformed_text=None,
        voice_backend=backend,
        verdict="skip",
        failure_reason="inference not implemented (skeleton)",
    )


# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------

def write_results(results: list[VoiceEvalResult], output_path: Path) -> None:
    """Write results to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    logger.info("wrote %d results to %s", len(results), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run voice eval cases")
    parser.add_argument("--backend", default="irodori", choices=["irodori", "openvoice", "f5-tts", "cosyvoice", "indextts", "rvc", "xtts"],
                        help="TTS backend to evaluate (default: irodori)")
    parser.add_argument("--out", default=str(RESULTS_DIR / "voice_eval_smoke.jsonl"),
                        help="Output JSONL path (default: evals/results/voice_eval_smoke.jsonl)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip real inference, validate fixture paths only")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    cases = load_cases()
    logger.info("loaded %d voice eval cases", len(cases))

    fixture_errors = validate_fixtures(cases)
    if fixture_errors:
        for cid, err in fixture_errors.items():
            logger.warning("[fixture] %s: %s", cid, err)

    results: list[VoiceEvalResult] = []
    for case in cases:
        result = process_case(case, backend=args.backend, dry_run=args.dry_run)
        results.append(result)

    output_path = Path(args.out)
    write_results(results, output_path)

    # Summary
    verdicts = [r.verdict for r in results]
    logger.info("results: total=%d pass=%d warn=%d fail=%d skip=%d",
                len(verdicts),
                verdicts.count("pass"),
                verdicts.count("warn"),
                verdicts.count("fail"),
                verdicts.count("skip"))


if __name__ == "__main__":
    main()
