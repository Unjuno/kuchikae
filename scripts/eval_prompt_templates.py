#!/usr/bin/env python3
"""Evaluation script for LLM prompt templates.

This script evaluates new LLM prompt templates by running them against
a set of test inputs and saving the results to artifacts.

Usage:
    uv run python scripts/eval_prompt_templates.py --template "先生っぽく"
    uv run python scripts/eval_prompt_templates.py --template "関西弁"
    uv run python scripts/eval_prompt_templates.py --all-new
    uv run python scripts/eval_prompt_templates.py --experimental-only
    uv run python scripts/eval_prompt_templates.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from kuchikae.ui.templates import TEMPLATES


# Test inputs for evaluation
TEST_INPUTS = [
    "こんにちは",
    "ありがとう",
    "ごめん、ちょっと遅れます",
    "明日15時に資料を送ってください",
    "田中さんに確認してください",
    "今日は参加できません",
    "がんばって",
    "これどう思う？",
    "その操作は危ないです",
    "今日はちょっと疲れた",
    "お疲れさまです",
    "無理です",
    "今日はもう帰ります",
    "6月20日に渋谷で打ち合わせです",
    "バグはまだ直っていません",
    "それは違うと思います",
]

# Official candidate templates
OFFICIAL_CANDIDATES = [
    "先生っぽく",
    "友達っぽく",
    "ニュースキャスターっぽく",
    "セールスっぽく",
    "詩的に",
]

# Experimental candidate templates
EXPERIMENTAL_CANDIDATES = [
    "実験: 関西弁",
    "実験: ギャルっぽく",
    "実験: 赤ちゃんっぽく",
    "実験: 武士っぽく",
    "実験: 毒舌",
    "実験: 皮肉っぽく",
    "実験: 外国人っぽく",
    "実験: 特定キャラっぽく",
]


def get_new_templates() -> list[str]:
    """Get all new templates (official + experimental)."""
    return OFFICIAL_CANDIDATES + EXPERIMENTAL_CANDIDATES


def transform_text(text: str, template_name: str) -> dict[str, Any]:
    """Transform text using the specified template via Ollama."""
    from kuchikae.domain.text_transform import OllamaTextTransformBackend
    from kuchikae.domain.types import TextTransformPrompt

    template_text = TEMPLATES.get(template_name, "")
    prompt = TextTransformPrompt(instruction=template_text)
    backend = OllamaTextTransformBackend()

    start_time = time.time()
    try:
        result = backend.transform(text, prompt)
        elapsed = time.time() - start_time
        return {
            "success": True,
            "output": result,
            "elapsed_seconds": elapsed,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "output": None,
            "elapsed_seconds": elapsed,
            "error": str(e),
        }


def evaluate_template(template_name: str, inputs: list[str]) -> list[dict[str, Any]]:
    """Evaluate a single template against all test inputs."""
    results = []
    for i, text in enumerate(inputs):
        print(f"  [{i+1}/{len(inputs)}] {text[:30]}...", end=" ", flush=True)
        result = transform_text(text, template_name)
        result["template"] = template_name
        result["input"] = text
        results.append(result)
        if result["success"]:
            print(f"✓ ({result['elapsed_seconds']:.1f}s)")
        else:
            print(f"✗ {result['error']}")
    return results


def save_jsonl(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save results to JSONL file."""
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def save_markdown(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save results to Markdown file."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# LLM Template Evaluation Results\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Group by template
        by_template: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            template = result["template"]
            if template not in by_template:
                by_template[template] = []
            by_template[template].append(result)

        for template, template_results in by_template.items():
            f.write(f"## {template}\n\n")
            for result in template_results:
                status = "✓" if result["success"] else "✗"
                f.write(f"### {status} Input: `{result['input']}`\n\n")
                if result["success"]:
                    f.write(f"**Output:** {result['output']}\n\n")
                    f.write(f"**Time:** {result['elapsed_seconds']:.1f}s\n\n")
                else:
                    f.write(f"**Error:** {result['error']}\n\n")
                f.write("---\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM prompt templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Evaluate a specific template",
    )
    parser.add_argument(
        "--all-new",
        action="store_true",
        help="Evaluate all new templates (official + experimental)",
    )
    parser.add_argument(
        "--experimental-only",
        action="store_true",
        help="Evaluate only experimental templates",
    )
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Evaluate only official candidate templates",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for results (default: artifacts)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        help="Version suffix for output files (default: v2)",
    )

    args = parser.parse_args()

    # Determine which templates to evaluate
    templates_to_evaluate = []
    if args.template:
        if args.template not in TEMPLATES:
            print(f"Error: Template '{args.template}' not found in TEMPLATES")
            print(f"Available templates: {list(TEMPLATES.keys())}")
            sys.exit(1)
        templates_to_evaluate = [args.template]
    elif args.all_new:
        templates_to_evaluate = get_new_templates()
    elif args.experimental_only:
        templates_to_evaluate = EXPERIMENTAL_CANDIDATES
    elif args.official_only:
        templates_to_evaluate = OFFICIAL_CANDIDATES
    else:
        parser.print_help()
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(templates_to_evaluate)} templates...")
    print(f"Templates: {templates_to_evaluate}")
    print(f"Test inputs: {len(TEST_INPUTS)}")
    print()

    all_results = []
    for template in templates_to_evaluate:
        print(f"Template: {template}")
        results = evaluate_template(template, TEST_INPUTS)
        all_results.extend(results)
        print()

    # Save results
    jsonl_path = output_dir / f"prompt_eval_results_{args.version}.jsonl"
    md_path = output_dir / f"prompt_eval_results_{args.version}.md"

    save_jsonl(all_results, jsonl_path)
    save_markdown(all_results, md_path)

    print(f"Results saved to:")
    print(f"  - {jsonl_path}")
    print(f"  - {md_path}")

    # Print summary
    success_count = sum(1 for r in all_results if r["success"])
    fail_count = len(all_results) - success_count
    print(f"\nSummary: {success_count} success, {fail_count} failed")


if __name__ == "__main__":
    main()
