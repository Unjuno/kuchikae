#!/usr/bin/env python3
"""Run voice/audio evaluation cases.

Usage:
  uv run python evals/run_voice_eval.py --dry-run
  uv run python evals/run_voice_eval.py --mode tts-only --backend irodori
  uv run python evals/run_voice_eval.py --mode pipeline --backend irodori
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
    mode: str = "tts-only"
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
            template=raw.get("template", u"自然に"),
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
# Voice backend factory
# ---------------------------------------------------------------------------


def _build_voice_backend(backend: str) -> Any:
    """Build a voice output backend by name.

    Raises NotImplementedError for backends that are not yet implemented
    in the eval harness.
    """
    if backend == "irodori":
        from kuchikae.backends.voice_output import IrodoriTTSVoiceOutputBackend
        return IrodoriTTSVoiceOutputBackend()
    elif backend == "openvoice":
        from kuchikae.backends.voice_output import OpenVoiceOutputBackend
        return OpenVoiceOutputBackend()
    else:
        raise NotImplementedError(
            f"Unsupported eval backend: {backend!r}. "
            f"Supported: irodori, openvoice"
        )


# ---------------------------------------------------------------------------
# Duration helper (sample-rate aware)
# ---------------------------------------------------------------------------


def _compute_duration_ratio(input_path: str, output_path: str) -> float | None:
    try:
        import soundfile as sf
        in_audio, in_sr = sf.read(input_path)
        out_audio, out_sr = sf.read(output_path)
        in_dur = len(in_audio) / in_sr
        out_dur = len(out_audio) / out_sr
        if in_dur <= 0:
            return None
        return out_dur / in_dur
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Text transform helper (for tts-only mode)
# ---------------------------------------------------------------------------


def _apply_text_transform(input_text: str, template_name: str) -> str | None:
    """Apply PromptedRuleTextTransformBackend to input_text using template_name.

    Returns transformed text, or None on failure.
    """
    try:
        from kuchikae.domain.text_transform import PromptedRuleTextTransformBackend
        from kuchikae.domain.types import TextTransformPrompt
        from kuchikae.ui.templates import TEMPLATES

        template_instruction = TEMPLATES.get(template_name, TEMPLATES[u"自然に"])
        backend = PromptedRuleTextTransformBackend()
        result = backend.transform(
            text=input_text,
            prompt=TextTransformPrompt(instruction=template_instruction),
        )
        if isinstance(result, str):
            return result
        # Some backends return (status, text) tuples
        if isinstance(result, tuple) and len(result) >= 2:
            return str(result[1])
        return str(result)
    except Exception:
        logger.exception("[text-transform] failed for template=%s", template_name)
        return None


# ---------------------------------------------------------------------------
# Process a single case
# ---------------------------------------------------------------------------


def process_case(
    case: VoiceEvalCase,
    pipeline: Any,
    backend: str,
    mode: str = "tts-only",
    dry_run: bool = False,
    voice_backend: Any = None,
) -> VoiceEvalResult:
    """Process a single voice eval case.

    In *tts-only* mode (default), STT is skipped: the ground-truth
    ``input_text`` is transformed via ``PromptedRuleTextTransformBackend``
    and fed directly to the voice backend.  The fixture WAV is used as
    reference audio only.

    In *pipeline* mode, the full E2E pipeline (STT → text-transform → TTS)
    is exercised.  A warning is emitted if dummy STT is detected.

    In *dry-run* mode, skips actual inference and returns a placeholder
    result.
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
            mode=mode,
            verdict="skip",
            failure_reason=f"fixture not found: {fixture_path}",
        )

    if dry_run:
        logger.info("[dry-run] case=%s template=%s fixture=%s mode=%s", case.id, case.template, fixture_path, mode)
        return VoiceEvalResult(
            case_id=case.id,
            input_audio=case.input_audio,
            output_audio=None,
            template=case.template,
            input_text=case.input_text,
            transformed_text=None,
            voice_backend=backend,
            mode=mode,
            verdict="skip",
            failure_reason="dry-run (no inference)",
        )

    # ── Real inference path ──
    try:
        t0 = time.time()

        if mode == "tts-only":
            transformed_text = _apply_text_transform(case.input_text, case.template)
            if transformed_text is None:
                return VoiceEvalResult(
                    case_id=case.id,
                    input_audio=case.input_audio,
                    output_audio=None,
                    template=case.template,
                    input_text=case.input_text,
                    transformed_text=None,
                    voice_backend=backend,
                    mode=mode,
                    verdict="fail",
                    failure_reason="text transform failed",
                )

            if "[DUMMY_STT_OUTPUT]" in transformed_text:
                return VoiceEvalResult(
                    case_id=case.id,
                    input_audio=case.input_audio,
                    output_audio=None,
                    template=case.template,
                    input_text=case.input_text,
                    transformed_text=transformed_text,
                    voice_backend=backend,
                    mode=mode,
                    verdict="fail",
                    failure_reason="transformed_text contains [DUMMY_STT_OUTPUT] (STT leak in tts-only mode)",
                )

            # ── Direct TTS call ──
            from kuchikae.domain.types import VoiceContext, VoiceOutputPrompt

            _vb = voice_backend if voice_backend is not None else _build_voice_backend(backend)
            voice_context = VoiceContext(
                reference_audio_path=str(fixture_path),
                ready=True,
            )
            voice_prompt = VoiceOutputPrompt(instruction=transformed_text)
            output_audio = _vb.synthesize(
                text=transformed_text,
                voice_context=voice_context,
                prompt=voice_prompt,
            )

            elapsed = time.time() - t0
            duration_ratio = _compute_duration_ratio(str(fixture_path), str(output_audio))

            logger.info(
                "[eval] case=%s elapsed=%.1fs trf=%s out=%s dur_ratio=%s",
                case.id, elapsed,
                transformed_text[:40],
                output_audio,
                f"{duration_ratio:.2f}" if duration_ratio else "N/A",
            )

            failure_reason = ""
            verdict: Verdict = "pass"
            if duration_ratio is not None:
                if duration_ratio > 4.0:
                    verdict = "warn"
                    failure_reason = f"duration_ratio too high: {duration_ratio:.2f}"
                elif duration_ratio < 0.3:
                    verdict = "warn"
                    failure_reason = f"duration_ratio too low: {duration_ratio:.2f}"

            return VoiceEvalResult(
                case_id=case.id,
                input_audio=case.input_audio,
                output_audio=str(output_audio),
                template=case.template,
                input_text=case.input_text,
                transformed_text=transformed_text,
                voice_backend=backend,
                mode=mode,
                duration_ratio=duration_ratio,
                verdict=verdict,
                failure_reason=failure_reason,
            )

        else:
            # ── Pipeline mode (E2E, uses STT) ──
            from kuchikae.domain.types import TextTransformPrompt
            from kuchikae.ui.templates import TEMPLATES

            template_text = TEMPLATES.get(case.template, TEMPLATES[u"自然に"])
            prompt = TextTransformPrompt(instruction=template_text)

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
                    mode=mode,
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
                    mode=mode,
                    verdict="fail",
                    failure_reason=f"pipeline did not complete (status={status})",
                )

            duration_ratio = _compute_duration_ratio(str(fixture_path), str(output_audio))

            # Warn if dummy STT leaked through
            warning = ""
            if transformed_text and "[DUMMY_STT_OUTPUT]" in transformed_text:
                warning = " (dummy STT detected – results not usable for voice quality assessment)"

            logger.info(
                "[eval] case=%s elapsed=%.1fs src=%s trf=%s out=%s dur_ratio=%s%s",
                case.id, elapsed,
                source_text[:40] if source_text else None,
                transformed_text[:40] if transformed_text else None,
                output_audio,
                f"{duration_ratio:.2f}" if duration_ratio else "N/A",
                warning,
            )

            failure_reason = warning.strip() if warning else ""
            verdict = "warn" if "[DUMMY_STT_OUTPUT]" in (transformed_text or "") else "pass"
            if duration_ratio is not None:
                if duration_ratio > 4.0:
                    verdict = "warn"
                    failure_reason = (failure_reason + "; " if failure_reason else "") + f"duration_ratio too high: {duration_ratio:.2f}"
                elif duration_ratio < 0.3:
                    verdict = "warn"
                    failure_reason = (failure_reason + "; " if failure_reason else "") + f"duration_ratio too low: {duration_ratio:.2f}"
            return VoiceEvalResult(
                case_id=case.id,
                input_audio=case.input_audio,
                output_audio=str(output_audio),
                template=case.template,
                input_text=case.input_text,
                transformed_text=transformed_text,
                voice_backend=backend,
                mode=mode,
                duration_ratio=duration_ratio,
                verdict=verdict,
                failure_reason=failure_reason,
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
            mode=mode,
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
    parser.add_argument("--mode", default="tts-only", choices=["tts-only", "pipeline"],
                        help="Evaluation mode (default: tts-only). "
                             "tts-only skips STT and uses ground-truth input_text; "
                             "pipeline runs full E2E including STT.")
    parser.add_argument("--backend", default="irodori", choices=["irodori", "openvoice"],
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

    pipeline = None
    shared_voice_backend = None
    if not args.dry_run:
        shared_voice_backend = _build_voice_backend(args.backend)
        if args.mode == "pipeline":
            from kuchikae.pipeline import KuchikaePipeline
            pipeline = KuchikaePipeline(
                voice_output_backend=shared_voice_backend,
                disable_processing_cache=True,
            )

    results: list[VoiceEvalResult] = []
    for case in cases:
        result = process_case(case, pipeline, backend=args.backend, mode=args.mode, dry_run=args.dry_run, voice_backend=shared_voice_backend)
        results.append(result)

    output_path = Path(args.out)
    write_results(results, output_path)

    # Summary
    verdicts = [r.verdict for r in results]
    logger.info("results: mode=%s total=%d pass=%d warn=%d fail=%d skip=%d",
                args.mode,
                len(verdicts),
                verdicts.count("pass"),
                verdicts.count("warn"),
                verdicts.count("fail"),
                verdicts.count("skip"))


if __name__ == "__main__":
    main()
