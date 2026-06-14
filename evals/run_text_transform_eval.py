#!/usr/bin/env python3
"""Run text transform evaluation cases.

Usage:
  uv run python evals/run_text_transform_eval.py --backend prompted_rule
  uv run python evals/run_text_transform_eval.py --backend ollama --ollama-model qwen2.5-coder:7b
  uv run python evals/run_text_transform_eval.py --template "実験: 関西弁"
  uv run python evals/run_text_transform_eval.py --backend prompted_rule --out evals/results/smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CASES_PATH = Path(__file__).parent / "text_transform_cases.yaml"
JUDGE_PROMPT_PATH = Path(__file__).parent / "judge_prompt.md"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class ExpectedResult:
    must_preserve: list[str] = field(default_factory=list)
    should_not_include: list[str] = field(default_factory=list)
    style_notes: str = ""
    failure_blockers: list[str] = field(default_factory=list)


@dataclass
class EvalCase:
    id: str
    category: str
    input: str
    template: str
    expected: ExpectedResult
    custom_prompt: str = ""


@dataclass
class RuleJudgeResult:
    overall_pass: bool
    failure_type: str | None
    failure_reason: str
    must_preserve_hits: dict[str, bool] = field(default_factory=dict)
    should_not_include_hits: dict[str, bool] = field(default_factory=dict)


@dataclass
class LLMJudgeResult:
    meaning_preservation: int = 0
    style_strength: int = 0
    naturalness: int = 0
    safety: int = 0
    overreach: int = 0
    overall_pass: bool = False
    failure_type: str | None = None
    failure_reason: str = ""


@dataclass
class EvalResult:
    case_id: str
    category: str
    template: str
    input_text: str
    transformed_text: str
    latency_ms: float
    rule_judge: RuleJudgeResult
    llm_judge: LLMJudgeResult | None = None


def load_cases(template_filter: str | None = None) -> list[EvalCase]:
    """Load evaluation cases from YAML."""
    with open(CASES_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cases: list[EvalCase] = []
    for raw in data.get("cases", []):
        exp_raw = raw.get("expected", {})
        expected = ExpectedResult(
            must_preserve=exp_raw.get("must_preserve", []),
            should_not_include=exp_raw.get("should_not_include", []),
            style_notes=exp_raw.get("style_notes", ""),
            failure_blockers=exp_raw.get("failure_blockers", []),
        )
        case = EvalCase(
            id=raw["id"],
            category=raw["category"],
            input=raw["input"],
            template=raw["template"],
            expected=expected,
            custom_prompt=raw.get("custom_prompt", ""),
        )
        if template_filter and case.template != template_filter:
            continue
        cases.append(case)

    return cases


def create_rule_judge() -> Any:
    """Create a PromptedRuleTextTransformBackend for rule-based evaluation."""
    from kuchikae.domain.text_transform import PromptedRuleTextTransformBackend

    return PromptedRuleTextTransformBackend()


def create_ollama_backend(model: str) -> Any:
    """Create an OllamaTextTransformBackend for LLM-based evaluation."""
    from kuchikae.domain.text_transform import OllamaTextTransformBackend

    return OllamaTextTransformBackend(model=model)


def run_transform(backend: Any, case: EvalCase, backend_type: str) -> tuple[str, float]:
    """Run transformation and return (output, latency_ms)."""
    from kuchikae.domain.types import TextTransformPrompt
    from kuchikae.ui.templates import TEMPLATES

    if case.template == "カスタム":
        instruction = case.custom_prompt
    else:
        instruction = TEMPLATES.get(case.template, TEMPLATES["自然に"])

    prompt = TextTransformPrompt(instruction=instruction)

    start = time.monotonic()
    try:
        result = backend.transform(case.input, prompt)
    except Exception as e:
        logger.warning("Transform failed for %s: %s", case.id, e)
        result = ""
    elapsed = (time.monotonic() - start) * 1000

    return result, elapsed


def judge_rule(case: EvalCase, output: str, backend_type: str = "prompted_rule") -> RuleJudgeResult:
    """Apply rule-based judging to a transform result."""
    # Check empty output
    if not output.strip():
        return RuleJudgeResult(
            overall_pass=False,
            failure_type="empty_output",
            failure_reason="Transform produced empty output",
        )

    # Check STYLE_TEMPLATE leak
    if "[STYLE_TEMPLATE:" in output:
        return RuleJudgeResult(
            overall_pass=False,
            failure_type="template_leak",
            failure_reason="Output contains [STYLE_TEMPLATE: ...] marker text",
        )

    # Check must_preserve
    must_preserve_hits: dict[str, bool] = {}
    for term in case.expected.must_preserve:
        must_preserve_hits[term] = term in output

    missing = [t for t, hit in must_preserve_hits.items() if not hit]
    if missing:
        return RuleJudgeResult(
            overall_pass=False,
            failure_type="meaning_loss",
            failure_reason=f"Missing required terms: {', '.join(missing)}",
            must_preserve_hits=must_preserve_hits,
        )

    # Check should_not_include (skip for rule backend - can't do safety filtering)
    should_not_include_hits: dict[str, bool] = {}
    if backend_type != "prompted_rule":
        for term in case.expected.should_not_include:
            should_not_include_hits[term] = term in output

        found_unwanted = [t for t, hit in should_not_include_hits.items() if hit]
        if found_unwanted:
            return RuleJudgeResult(
                overall_pass=False,
                failure_type="unsafe",
                failure_reason=f"Contains unwanted terms: {', '.join(found_unwanted)}",
                should_not_include_hits=should_not_include_hits,
            )

    # Check echo (output matches input exactly - no transformation)
    if output.strip() == case.input.strip():
        return RuleJudgeResult(
            overall_pass=False,
            failure_type="style_weak",
            failure_reason="Output is identical to input (no transformation applied)",
            must_preserve_hits=must_preserve_hits,
            should_not_include_hits=should_not_include_hits,
        )

    return RuleJudgeResult(
        overall_pass=True,
        failure_type=None,
        failure_reason="",
        must_preserve_hits=must_preserve_hits,
        should_not_include_hits=should_not_include_hits,
    )


def judge_llm(case: EvalCase, output: str, ollama_model: str) -> LLMJudgeResult | None:
    """Apply LLM-based judging using an Ollama model as judge."""
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed, skipping LLM judge")
        return None

    judge_prompt_text = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")

    user_message = f"""## Original text
{case.input}

## Transformation template
{case.template}

## Custom prompt
{case.custom_prompt or '(none)'}

## Transformed text
{output}

## Your evaluation
Return ONLY a JSON object."""

    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": judge_prompt_text},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post("http://localhost:11434/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")

            # Extract JSON from response
            import re

            json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in judge response for %s", case.id)
                return None

            scores = json.loads(json_match.group())
            return LLMJudgeResult(
                meaning_preservation=scores.get("meaning_preservation", 0),
                style_strength=scores.get("style_strength", 0),
                naturalness=scores.get("naturalness", 0),
                safety=scores.get("safety", 0),
                overreach=scores.get("overreach", 0),
                overall_pass=scores.get("overall_pass", False),
                failure_type=scores.get("failure_type"),
                failure_reason=scores.get("failure_reason", ""),
            )
    except Exception as e:
        logger.warning("LLM judge failed for %s: %s", case.id, e)
        return None


def write_jsonl(results: list[EvalResult], output_path: Path) -> None:
    """Write results as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            record = {
                "case_id": r.case_id,
                "category": r.category,
                "template": r.template,
                "input": r.input_text,
                "output": r.transformed_text,
                "latency_ms": round(r.latency_ms, 1),
                "rule_judge": asdict(r.rule_judge),
            }
            if r.llm_judge:
                record["llm_judge"] = asdict(r.llm_judge)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text transform evaluation")
    parser.add_argument(
        "--backend",
        choices=["prompted_rule", "ollama"],
        default="prompted_rule",
        help="Backend to evaluate (default: prompted_rule)",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen2.5-coder:7b",
        help="Ollama model name (when --backend ollama)",
    )
    parser.add_argument(
        "--judge",
        choices=["none", "ollama"],
        default="none",
        help="LLM judge backend (default: none = rule-only)",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen2.5-coder:7b",
        help="Ollama model for LLM judge",
    )
    parser.add_argument("--template", default=None, help="Filter to a single template")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (default: evals/results/<timestamp>.jsonl)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cases = load_cases(template_filter=args.template)
    if not cases:
        logger.error("No cases found")
        sys.exit(1)

    logger.info("Loaded %d evaluation cases", len(cases))

    # Create backend
    if args.backend == "prompted_rule":
        backend = create_rule_judge()
    else:
        backend = create_ollama_backend(args.ollama_model)

    # Run evaluation
    results: list[EvalResult] = []
    for i, case in enumerate(cases, 1):
        logger.info("[%d/%d] %s (%s)", i, len(cases), case.id, case.template)

        output, latency = run_transform(backend, case, args.backend)
        rule_result = judge_rule(case, output, args.backend)

        llm_result = None
        if args.judge == "ollama":
            llm_result = judge_llm(case, output, args.judge_model)

        results.append(
            EvalResult(
                case_id=case.id,
                category=case.category,
                template=case.template,
                input_text=case.input,
                transformed_text=output,
                latency_ms=latency,
                rule_judge=rule_result,
                llm_judge=llm_result,
            )
        )

        status = "PASS" if rule_result.overall_pass else f"FAIL({rule_result.failure_type})"
        logger.info("  -> %s (%.0fms) %s", status, latency, rule_result.failure_reason)

    # Write results
    if args.out:
        out_path = Path(args.out)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"{args.backend}_{ts}.jsonl"

    write_jsonl(results, out_path)
    logger.info("Results written to %s", out_path)

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.rule_judge.overall_pass)
    failed = total - passed
    logger.info("Summary: %d/%d passed (%d failed)", passed, total, failed)

    if failed > 0:
        logger.info("Failures:")
        for r in results:
            if not r.rule_judge.overall_pass:
                logger.info(
                    "  %s: %s - %s",
                    r.rule_judge.failure_type,
                    r.case_id,
                    r.rule_judge.failure_reason,
                )

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
