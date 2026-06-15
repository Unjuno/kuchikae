#!/usr/bin/env python3
"""Summarize voice evaluation results from JSONL.

Usage:
  uv run python evals/summarize_voice_eval.py evals/results/voice_eval_smoke.jsonl
  uv run python evals/summarize_voice_eval.py evals/results/voice_eval_smoke.jsonl --format markdown
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
    skipped: int = 0

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0


def load_results(jsonl_path: Path) -> list[dict]:
    results: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_summary(results: list[dict]) -> dict:
    overall = SummaryStats(total=len(results))
    by_backend: dict[str, SummaryStats] = defaultdict(SummaryStats)
    by_template: dict[str, SummaryStats] = defaultdict(SummaryStats)
    by_mode: dict[str, SummaryStats] = defaultdict(SummaryStats)
    speaker_similarities: list[float] = []
    duration_ratios: list[float] = []
    worst_cases: list[dict] = []

    for r in results:
        verdict = r.get("verdict", "skip")
        backend = r.get("voice_backend", "unknown")
        template = r.get("template", "unknown")
        mode = r.get("mode", "unknown")

        by_backend[backend].total += 1
        by_template[template].total += 1
        by_mode[mode].total += 1

        if verdict == "pass":
            overall.passed += 1
            by_backend[backend].passed += 1
            by_template[template].passed += 1
            by_mode[mode].passed += 1
        elif verdict == "warn":
            overall.warned += 1
            by_backend[backend].warned += 1
            by_template[template].warned += 1
            by_mode[mode].warned += 1
        elif verdict == "fail":
            overall.failed += 1
            by_backend[backend].failed += 1
            by_template[template].failed += 1
            by_mode[mode].failed += 1
            worst_cases.append(r)
        else:
            overall.skipped += 1
            by_backend[backend].skipped += 1
            by_template[template].skipped += 1
            by_mode[mode].skipped += 1

        ss = r.get("speaker_similarity")
        if ss is not None:
            speaker_similarities.append(float(ss))

        dr = r.get("duration_ratio")
        if dr is not None:
            duration_ratios.append(float(dr))

    avg_ss = (sum(speaker_similarities) / len(speaker_similarities)) if speaker_similarities else None
    avg_dr = (sum(duration_ratios) / len(duration_ratios)) if duration_ratios else None

    worst_cases.sort(key=lambda x: x.get("speaker_similarity") if x.get("speaker_similarity") is not None else 1.0)

    return {
        "overall": {
            "total": overall.total,
            "passed": overall.passed,
            "warned": overall.warned,
            "failed": overall.failed,
            "skipped": overall.skipped,
            "pass_rate": overall.pass_rate,
        },
        "by_backend": {
            b: {
                "total": s.total,
                "passed": s.passed,
                "warned": s.warned,
                "failed": s.failed,
                "skipped": s.skipped,
            }
            for b, s in sorted(by_backend.items())
        },
        "by_template": {
            t: {
                "total": s.total,
                "passed": s.passed,
                "warned": s.warned,
                "failed": s.failed,
                "skipped": s.skipped,
            }
            for t, s in sorted(by_template.items())
        },
        "by_mode": {
            m: {
                "total": s.total,
                "passed": s.passed,
                "warned": s.warned,
                "failed": s.failed,
                "skipped": s.skipped,
            }
            for m, s in sorted(by_mode.items())
        },
        "average_speaker_similarity": avg_ss,
        "average_duration_ratio": avg_dr,
        "worst_cases": [
            {
                "case_id": c.get("case_id"),
                "speaker_similarity": c.get("speaker_similarity"),
                "voice_backend": c.get("voice_backend"),
                "failure_reason": c.get("failure_reason"),
            }
            for c in worst_cases[:10]
        ],
    }


def format_text(summary: dict) -> str:
    o = summary["overall"]
    lines = [
        f"Total: {o['total']}",
        f"Pass:  {o['passed']} ({o['pass_rate']:.1f}%)",
        f"Warn:  {o['warned']}",
        f"Fail:  {o['failed']}",
        f"Skip:  {o['skipped']}",
        "",
    ]
    if summary["average_speaker_similarity"] is not None:
        lines.append(f"Avg speaker similarity: {summary['average_speaker_similarity']:.3f}")
    if summary["average_duration_ratio"] is not None:
        lines.append(f"Avg duration ratio:     {summary['average_duration_ratio']:.3f}")
    lines.append("")

    if summary.get("by_backend"):
        lines.append("--- by backend ---")
        for backend, s in summary["by_backend"].items():
            lines.append(f"  {backend}: {s['total']} total, {s['passed']} pass, {s['failed']} fail, {s['skipped']} skip")
        lines.append("")

    if summary.get("by_mode"):
        lines.append("--- by mode ---")
        for mode, s in summary["by_mode"].items():
            lines.append(f"  {mode}: {s['total']} total, {s['passed']} pass, {s['failed']} fail, {s['skipped']} skip")
        lines.append("")

    if summary["worst_cases"]:
        lines.append("--- worst cases ---")
        for c in summary["worst_cases"]:
            lines.append(f"  {c['case_id']}: ss={c['speaker_similarity']} backend={c['voice_backend']} reason={c['failure_reason']}")
        lines.append("")

    return "\n".join(lines)


def format_markdown(summary: dict) -> str:
    o = summary["overall"]
    lines = [
        "## Voice eval summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total | {o['total']} |",
        f"| Pass | {o['passed']} ({o['pass_rate']:.1f}%) |",
        f"| Warn | {o['warned']} |",
        f"| Fail | {o['failed']} |",
        f"| Skip | {o['skipped']} |",
    ]
    if summary["average_speaker_similarity"] is not None:
        lines.append(f"| Avg speaker similarity | {summary['average_speaker_similarity']:.3f} |")
    if summary["average_duration_ratio"] is not None:
        lines.append(f"| Avg duration ratio | {summary['average_duration_ratio']:.3f} |")
    lines.append("")

    if summary.get("by_backend"):
        lines.append("### By backend")
        lines.append("")
        lines.append("| Backend | Total | Pass | Fail | Skip |")
        lines.append("|---------|-------|------|------|------|")
        for backend, s in summary["by_backend"].items():
            lines.append(f"| {backend} | {s['total']} | {s['passed']} | {s['failed']} | {s['skipped']} |")
        lines.append("")

    if summary.get("by_mode"):
        lines.append("### By mode")
        lines.append("")
        lines.append("| Mode | Total | Pass | Fail | Skip |")
        lines.append("|------|-------|------|------|------|")
        for mode, s in summary["by_mode"].items():
            lines.append(f"| {mode} | {s['total']} | {s['passed']} | {s['failed']} | {s['skipped']} |")
        lines.append("")

    if summary["worst_cases"]:
        lines.append("### Worst cases")
        lines.append("")
        lines.append("| Case ID | Speaker Similarity | Backend | Reason |")
        lines.append("|---------|-------------------|---------|--------|")
        for c in summary["worst_cases"]:
            lines.append(f"| {c['case_id']} | {c['speaker_similarity']} | {c['voice_backend']} | {c['failure_reason']} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize voice eval results")
    parser.add_argument("jsonl", type=Path, help="Path to JSONL results file")
    parser.add_argument("--format", choices=["text", "markdown"], default="text",
                        help="Output format (default: text)")
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"Error: {args.jsonl} not found", file=sys.stderr)
        sys.exit(1)

    results = load_results(args.jsonl)
    if not results:
        print("No results found.")
        sys.exit(0)

    summary = compute_summary(results)

    if args.format == "markdown":
        print(format_markdown(summary))
    else:
        print(format_text(summary))


if __name__ == "__main__":
    main()
