#!/usr/bin/env python3
"""Summarize text transform evaluation results from JSONL.

Usage:
  uv run python evals/summarize_eval.py evals/results/smoke.jsonl
  uv run python evals/summarize_eval.py evals/results/smoke.jsonl --format markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SummaryStats:
    total: int = 0
    passed: int = 0
    warned: int = 0
    failed: int = 0

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0

    @property
    def fail_rate(self) -> float:
        return (self.failed / self.total * 100) if self.total > 0 else 0.0


def load_results(jsonl_path: Path) -> list[dict]:
    """Load results from JSONL file."""
    results = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _get_verdict(rule: dict) -> str:
    return rule.get("verdict", "pass" if rule.get("overall_pass", False) else "fail")


def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics from results."""
    overall = SummaryStats(total=len(results))
    by_template: dict[str, SummaryStats] = defaultdict(SummaryStats)
    by_category: dict[str, SummaryStats] = defaultdict(SummaryStats)
    failure_types: dict[str, int] = defaultdict(int)
    blockers: list[dict] = []

    for r in results:
        rule = r.get("rule_judge", {})
        verdict = _get_verdict(rule)
        template = r.get("template", "unknown")
        category = r.get("category", "unknown")
        failure_type = rule.get("failure_type")

        by_template[template].total += 1
        by_category[category].total += 1

        if verdict == "pass":
            overall.passed += 1
            by_template[template].passed += 1
            by_category[category].passed += 1
        elif verdict == "warn":
            overall.warned += 1
            by_template[template].warned += 1
            by_category[category].warned += 1
        else:
            overall.failed += 1
            by_template[template].failed += 1
            by_category[category].failed += 1
            if failure_type:
                failure_types[failure_type] += 1
                blockers.append(
                    {
                        "case_id": r.get("case_id", ""),
                        "template": template,
                        "category": category,
                        "failure_type": failure_type,
                        "failure_reason": rule.get("failure_reason", ""),
                        "output": r.get("output", "")[:100],
                    }
                )

    return {
        "overall": {
            "total": overall.total,
            "passed": overall.passed,
            "warned": overall.warned,
            "failed": overall.failed,
            "pass_rate": round(overall.pass_rate, 1),
        },
        "by_template": {
            t: {
                "total": s.total,
                "passed": s.passed,
                "warned": s.warned,
                "failed": s.failed,
                "pass_rate": round(s.pass_rate, 1),
            }
            for t, s in sorted(by_template.items())
        },
        "by_category": {
            c: {
                "total": s.total,
                "passed": s.passed,
                "warned": s.warned,
                "failed": s.failed,
                "pass_rate": round(s.pass_rate, 1),
            }
            for c, s in sorted(by_category.items())
        },
        "failure_types": dict(sorted(failure_types.items(), key=lambda x: -x[1])),
        "blockers": blockers,
    }


def format_text(summary: dict) -> str:
    """Format summary as plain text."""
    lines = []
    o = summary["overall"]
    lines.append(f"Overall: {o['passed']}/{o['total']} passed ({o['pass_rate']}%), {o['warned']} warned, {o['failed']} failed")
    lines.append("")

    lines.append("By Template:")
    for t, s in summary["by_template"].items():
        status = "OK" if s["failed"] == 0 else f"{s['failed']} FAIL"
        if s["warned"] > 0:
            status += f" {s['warned']} WARN"
        lines.append(f"  {t}: {s['passed']}/{s['total']} ({s['pass_rate']}%) [{status}]")
    lines.append("")

    lines.append("By Category:")
    for c, s in summary["by_category"].items():
        status = "OK" if s["failed"] == 0 else f"{s['failed']} FAIL"
        if s["warned"] > 0:
            status += f" {s['warned']} WARN"
        lines.append(f"  {c}: {s['passed']}/{s['total']} ({s['pass_rate']}%) [{status}]")
    lines.append("")

    if summary["failure_types"]:
        lines.append("Failure Types:")
        for ft, count in summary["failure_types"].items():
            lines.append(f"  {ft}: {count}")
        lines.append("")

    if summary["blockers"]:
        lines.append("Blockers:")
        for b in summary["blockers"]:
            lines.append(
                f"  [{b['failure_type']}] {b['case_id']} ({b['template']}): {b['failure_reason']}"
            )
            if b["output"]:
                lines.append(f"    output: {b['output'][:80]}...")
        lines.append("")

    return "\n".join(lines)


def format_markdown(summary: dict) -> str:
    """Format summary as markdown."""
    lines = []
    o = summary["overall"]
    lines.append("# Text Transform Evaluation Results")
    lines.append("")
    lines.append(f"**Overall: {o['passed']}/{o['total']} passed ({o['pass_rate']}%), {o['warned']} warned, {o['failed']} failed**")
    lines.append("")

    lines.append("## By Template")
    lines.append("")
    lines.append("| Template | Pass | Warn | Fail | Total | Rate |")
    lines.append("|----------|------|------|------|-------|------|")
    for t, s in summary["by_template"].items():
        lines.append(f"| {t} | {s['passed']} | {s['warned']} | {s['failed']} | {s['total']} | {s['pass_rate']}% |")
    lines.append("")

    lines.append("## By Category")
    lines.append("")
    lines.append("| Category | Pass | Warn | Fail | Total | Rate |")
    lines.append("|----------|------|------|------|-------|------|")
    for c, s in summary["by_category"].items():
        lines.append(f"| {c} | {s['passed']} | {s['warned']} | {s['failed']} | {s['total']} | {s['pass_rate']}% |")
    lines.append("")

    if summary["failure_types"]:
        lines.append("## Failure Types")
        lines.append("")
        for ft, count in summary["failure_types"].items():
            lines.append(f"- **{ft}**: {count}")
        lines.append("")

    if summary["blockers"]:
        lines.append("## Blockers")
        lines.append("")
        for b in summary["blockers"]:
            lines.append(
                f"- **[{b['failure_type']}]** `{b['case_id']}` ({b['template']}): {b['failure_reason']}"
            )
            if b["output"]:
                lines.append(f"  - output: `{b['output'][:80]}...`")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("jsonl_path", help="Path to JSONL results file")
    parser.add_argument(
        "--format",
        choices=["text", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    path = Path(args.jsonl_path)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    results = load_results(path)
    if not results:
        print("No results found", file=sys.stderr)
        sys.exit(1)

    summary = compute_summary(results)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.format == "markdown":
        print(format_markdown(summary))
    else:
        print(format_text(summary))


if __name__ == "__main__":
    main()
