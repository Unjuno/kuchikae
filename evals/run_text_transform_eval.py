#!/usr/bin/env python3
"""Run text transform evaluation cases.

Usage:
  uv run python evals/run_text_transform_eval.py --backend prompted_rule
  uv run python evals/run_text_transform_eval.py --backend ollama
  uv run python evals/run_text_transform_eval.py --template "実験: 関西弁"
  uv run python evals/run_text_transform_eval.py --backend prompted_rule --out evals/results/smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CASES_PATH = Path(__file__).parent / "text_transform_cases.yaml"
JUDGE_PROMPT_PATH = Path(__file__).parent / "judge_prompt.md"
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Schema: semantic_preserve entry
# ---------------------------------------------------------------------------

@dataclass
class SemanticPreserveEntry:
    canonical: str
    allowed: list[str]

    @classmethod
    def from_raw(cls, raw: str | dict) -> SemanticPreserveEntry:
        if isinstance(raw, str):
            return cls(canonical=raw, allowed=[raw])
        return cls(
            canonical=raw.get("canonical", ""),
            allowed=raw.get("allowed", []),
        )


# ---------------------------------------------------------------------------
# Schema: expected result
# ---------------------------------------------------------------------------

@dataclass
class ExpectedResult:
    hard_preserve: list[str] = field(default_factory=list)
    semantic_preserve: list[SemanticPreserveEntry] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    style_notes: str = ""
    # backward compat
    must_preserve: list[str] = field(default_factory=list)
    should_not_include: list[str] = field(default_factory=list)
    failure_blockers: list[str] = field(default_factory=list)

    def normalized_hard(self) -> list[str]:
        return self.hard_preserve or self.must_preserve

    def normalized_forbidden(self) -> list[str]:
        return self.forbidden or self.should_not_include

    def normalized_blockers(self) -> list[str]:
        return self.blockers or self.failure_blockers


# ---------------------------------------------------------------------------
# Eval case
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    id: str
    category: str
    input: str
    template: str
    expected: ExpectedResult
    custom_prompt: str = ""


# ---------------------------------------------------------------------------
# Judge results
# ---------------------------------------------------------------------------

Verdict = Literal["pass", "warn", "fail"]


@dataclass
class RuleJudgeResult:
    verdict: Verdict
    hard_preserve_hits: dict[str, bool] = field(default_factory=dict)
    semantic_preserve_hits: dict[str, bool] = field(default_factory=dict)
    forbidden_hits: dict[str, bool] = field(default_factory=dict)
    failure_type: str | None = None
    failure_reason: str = ""

    @property
    def overall_pass(self) -> bool:
        return self.verdict != "fail"


@dataclass
class LLMJudgeResult:
    meaning_preservation: int = 0
    style_strength: int = 0
    naturalness: int = 0
    safety: int = 0
    overreach: int = 0
    verdict: Verdict = "pass"
    failure_type: str | None = None
    failure_reason: str = ""

    @property
    def overall_pass(self) -> bool:
        return self.verdict != "fail"


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


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _parse_semantic(raw_list: list) -> list[SemanticPreserveEntry]:
    entries: list[SemanticPreserveEntry] = []
    for item in raw_list:
        entries.append(SemanticPreserveEntry.from_raw(item))
    return entries


def load_cases(template_filter: str | None = None) -> list[EvalCase]:
    """Load evaluation cases from YAML."""
    with open(CASES_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cases: list[EvalCase] = []
    for raw in data.get("cases", []):
        exp_raw = raw.get("expected", {})
        expected = ExpectedResult(
            hard_preserve=exp_raw.get("hard_preserve", []),
            semantic_preserve=_parse_semantic(exp_raw.get("semantic_preserve", [])),
            forbidden=exp_raw.get("forbidden", []),
            blockers=exp_raw.get("blockers", []),
            style_notes=exp_raw.get("style_notes", ""),
            # backward compat
            must_preserve=exp_raw.get("must_preserve", []),
            should_not_include=exp_raw.get("should_not_include", []),
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


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Rule judge — pass / warn / fail
# ---------------------------------------------------------------------------

def judge_rule(case: EvalCase, output: str, backend_type: str = "prompted_rule") -> RuleJudgeResult:
    """Apply rule-based judging. Returns pass/warn/fail verdict."""
    exp = case.expected
    hard = exp.normalized_hard()
    forbidden = exp.normalized_forbidden()
    sem_entries = exp.semantic_preserve

    # ── fail: empty output ──
    if not output.strip():
        return RuleJudgeResult(
            verdict="fail",
            failure_type="empty_output",
            failure_reason="Transform produced empty output",
        )

    # ── fail: STYLE_TEMPLATE leak ──
    if "[STYLE_TEMPLATE:" in output:
        return RuleJudgeResult(
            verdict="fail",
            failure_type="template_leak",
            failure_reason="Output contains [STYLE_TEMPLATE: ...] marker text",
        )

    # ── fail: CoT leak (case-insensitive) ──
    lower_out = output.lower()
    if "<think>" in lower_out or "</think>" in lower_out or "<thought>" in lower_out or "</thought>" in lower_out:
        return RuleJudgeResult(
            verdict="fail",
            failure_type="cot_leak",
            failure_reason="Output contains CoT tag (<think>/</thought>/<thought>)",
        )

    # ── hard_preserve: fail on missing ──
    hard_hits: dict[str, bool] = {}
    for term in hard:
        hard_hits[term] = term in output
    missing_hard = [t for t, hit in hard_hits.items() if not hit]
    if missing_hard:
        return RuleJudgeResult(
            verdict="fail",
            failure_type="meaning_loss",
            failure_reason=f"Missing required terms: {', '.join(missing_hard)}",
            hard_preserve_hits=hard_hits,
        )

    # ── forbidden: fail on hit (always run — string check) ──
    forbidden_hits: dict[str, bool] = {}
    for term in forbidden:
        forbidden_hits[term] = term in output
    found_forbidden = [t for t, hit in forbidden_hits.items() if hit]
    if found_forbidden:
        return RuleJudgeResult(
            verdict="fail",
            failure_type="unsafe",
            failure_reason=f"Contains forbidden terms: {', '.join(found_forbidden)}",
            forbidden_hits=forbidden_hits,
        )

    # ── semantic_preserve: warn on missing ──
    sem_hits: dict[str, bool] = {}
    sem_warns: list[str] = []
    for entry in sem_entries:
        hit = any(a in output for a in entry.allowed)
        sem_hits[entry.canonical] = hit
        if not hit:
            sem_warns.append(entry.canonical)

    # ── echo: no transformation ──
    if output.strip() == case.input.strip():
        return RuleJudgeResult(
            verdict="fail",
            failure_type="style_weak",
            failure_reason="Output is identical to input (no transformation applied)",
            hard_preserve_hits=hard_hits,
            semantic_preserve_hits=sem_hits,
            forbidden_hits=forbidden_hits,
        )

    # ── verdict ──
    if sem_warns:
        return RuleJudgeResult(
            verdict="warn",
            failure_type="semantic_loss",
            failure_reason=f"Semantic terms not found: {', '.join(sem_warns)}",
            hard_preserve_hits=hard_hits,
            semantic_preserve_hits=sem_hits,
            forbidden_hits=forbidden_hits,
        )

    return RuleJudgeResult(
        verdict="pass",
        hard_preserve_hits=hard_hits,
        semantic_preserve_hits=sem_hits,
        forbidden_hits=forbidden_hits,
    )


# ---------------------------------------------------------------------------
# LLM judge — only for warn / ambiguous semantic cases
# ---------------------------------------------------------------------------

def judge_llm(case: EvalCase, output: str, ollama_model: str) -> LLMJudgeResult | None:
    """Apply LLM-based judging for semantic/ambiguous cases."""
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

            json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in judge response for %s", case.id)
                return None

            scores = json.loads(json_match.group())
            # Verdict: prefer explicit verdict field, fallback to overall_pass
            raw_verdict = scores.get("verdict")
            if raw_verdict in ("pass", "warn", "fail"):
                verdict = raw_verdict
            else:
                verdict = "pass" if scores.get("overall_pass", False) else "warn"
            return LLMJudgeResult(
                meaning_preservation=scores.get("meaning_preservation", 0),
                style_strength=scores.get("style_strength", 0),
                naturalness=scores.get("naturalness", 0),
                safety=scores.get("safety", 0),
                overreach=scores.get("overreach", 0),
                verdict=verdict,
                failure_type=scores.get("failure_type"),
                failure_reason=scores.get("failure_reason", ""),
            )
    except Exception as e:
        logger.warning("LLM judge failed for %s: %s", case.id, e)
        return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(results: list[EvalResult], output_path: Path, model_name: str = "") -> None:
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
            if model_name:
                record["model"] = model_name
            if r.llm_judge:
                record["llm_judge"] = asdict(r.llm_judge)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        default=None,
        help="Ollama model name (when --backend ollama). Default: qwen2.5:7b-instruct",
    )
    parser.add_argument(
        "--judge",
        choices=["none", "ollama"],
        default="none",
        help="LLM judge backend (default: none = rule-only). Used for semantic/warn cases.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Ollama model for LLM judge. Default: same as --ollama-model",
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

    # Resolve default model
    from kuchikae.domain.text_transform import DEFAULT_OLLAMA_TEXT_MODEL

    if args.ollama_model is None:
        args.ollama_model = DEFAULT_OLLAMA_TEXT_MODEL
    if args.judge_model is None:
        args.judge_model = args.ollama_model

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

        # LLM judge: only for warn verdict (semantic ambiguity)
        llm_result = None
        if args.judge == "ollama" and rule_result.verdict == "warn":
            logger.info("  -> warn: invoking LLM judge for semantic check")
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

        verdict = rule_result.verdict.upper()
        reason = rule_result.failure_reason
        logger.info("  -> %s (%.0fms) %s", verdict, latency, reason)

    # Write results
    if args.out:
        out_path = Path(args.out)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"{args.backend}_{ts}.jsonl"

    write_jsonl(results, out_path, model_name=args.ollama_model)
    logger.info("Results written to %s", out_path)

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.rule_judge.verdict == "pass")
    warned = sum(1 for r in results if r.rule_judge.verdict == "warn")
    failed = sum(1 for r in results if r.rule_judge.verdict == "fail")
    logger.info("Summary: %d pass, %d warn, %d fail (total %d)", passed, warned, failed, total)

    if failed > 0:
        logger.info("Failures:")
        for r in results:
            if r.rule_judge.verdict == "fail":
                logger.info(
                    "  %s: %s - %s",
                    r.rule_judge.failure_type,
                    r.case_id,
                    r.rule_judge.failure_reason,
                )

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
