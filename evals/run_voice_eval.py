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
import os
import sys
import time
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
# Pipeline helpers
# ---------------------------------------------------------------------------


def _build_pipeline(backend: str) -> Any:
    """Build a KuchikaePipeline for the given backend."""
    from kuchikae.pipeline import KuchikaePipeline
    from kuchikae.backends.voice_output import (
        IrodoriTTSVoiceOutputBackend,
    )

    if backend == "irodori":
        voice_backend = IrodoriTTSVoiceOutputBackend()
    else:
        raise ValueError(f"unsupported backend for eval: {backend}")

    return KuchikaePipeline(
        voice_output_backend=voice_backend,
        disable_processing_cache=True,
    )


def _compute_duration_ratio(input_path: str, output_path: str) -> float | None:
    try:
        import soundfile as sf
        in_dur = len(sf.read(input_path)[0]) / 16000.0
        out_dur = len(sf.read(output_path)[0]) / 16000.0
        if in_dur <= 0:
            return None
        return out_dur / in_dur
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Process a single case
# ---------------------------------------------------------------------------


def process_case(
    case: VoiceEvalCase,
    pipeline: Any,
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

    # ── Real inference path ──
    try:
        from kuchikae.domain.types import TextTransformPrompt
        from kuchikae.ui.templates import TEMPLATES

        template_text = TEMPLATES.get(case.template, TEMPLATES["自然に"])
        prompt = TextTransformPrompt(instruction=template_text)
        t0 = time.time()

        # process_stream returns (status, source, transformed, audio)
        result_gen = pipeline.process_stream(str(fixture_path), prompt, voice_style="auto")
        last_result: tuple[str, str | None, str | None, str | None] | None = None
        for result in result_gen:
            last_result = result
        elapsed = time.time() - t0

        if last_result is None:
            return VoiceEvalResult(
                case_id=case.id,
                input_audio=case.input_audio,
                output_audio=None,
                template=case.template,
                input_text=case.input_text,
                transformed_text=None,
                voice_backend=backend,
                verdict="fail",
                failure_reason="pipeline returned no results",
            )

        status, source_text, transformed_text, output_audio = last_result
        if status != "DONE" or not output_audio:
            return VoiceEvalResult(
                case_id=case.id,
                input_audio=case.input_audio,
                output_audio=None,
                template=case.template,
                input_text=case.input_text,
                transformed_text=transformed_text,
                voice_backend=backend,
                verdict="fail",
                failure_reason=f"pipeline did not complete (status={status})",
            )

        duration_ratio = _compute_duration_ratio(str(fixture_path), str(output_audio))

        logger.info(
            "[eval] case=%s elapsed=%.1fs src=%s trf=%s out=%s dur_ratio=%s",
            case.id, elapsed,
            source_text[:40] if source_text else None,
            transformed_text[:40] if transformed_text else None,
            output_audio,
            f"{duration_ratio:.2f}" if duration_ratio else "N/A",
        )

        # Basic verdict: if the pipeline ran and produced output, pass
        # (full acoustic metric computation requires librosa/pyworld)
        verdict: Verdict = "pass"
        if duration_ratio is not None and (duration_ratio < 0.3 or duration_ratio > 4.0):
            verdict = "warn"

        return VoiceEvalResult(
            case_id=case.id,
            input_audio=case.input_audio,
            output_audio=str(output_audio),
            template=case.template,
            input_text=case.input_text,
            transformed_text=transformed_text,
            voice_backend=backend,
            duration_ratio=duration_ratio,
            verdict=verdict,
        )

    except Exception as e:
        logger.exception("[eval] case=%s failed", case.id)
        return VoiceEvalResult(
            case_id=case.id,
            input_audio=case.input_audio,
            output_audio=None,
            template=case.template,
            input_text=case.input_text,
            transformed_text=None,
            voice_backend=backend,
            verdict="fail",
            failure_reason=f"{type(e).__name__}: {e}",
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

    pipeline = _build_pipeline(args.backend) if not args.dry_run else None

    results: list[VoiceEvalResult] = []
    for case in cases:
        result = process_case(case, pipeline, backend=args.backend, dry_run=args.dry_run)
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
