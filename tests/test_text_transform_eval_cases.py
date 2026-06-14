"""Tests for the text transform evaluation dataset and scripts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kuchikae.ui.templates import TEMPLATES


EVALS_DIR = Path(__file__).parent.parent / "evals"
CASES_PATH = EVALS_DIR / "text_transform_cases.yaml"
RUNNER_PATH = EVALS_DIR / "run_text_transform_eval.py"
SUMMARIZER_PATH = EVALS_DIR / "summarize_eval.py"
JUDGE_PROMPT_PATH = EVALS_DIR / "judge_prompt.md"


# ---------------------------------------------------------------------------
# YAML structure tests
# ---------------------------------------------------------------------------


def _load_cases() -> list[dict]:
    with open(CASES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f).get("cases", [])


class TestEvalCasesStructure:
    def test_yaml_exists(self) -> None:
        assert CASES_PATH.exists(), f"YAML not found at {CASES_PATH}"

    def test_cases_are_list(self) -> None:
        data = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
        assert "cases" in data
        assert isinstance(data["cases"], list)

    def test_minimum_case_count(self) -> None:
        cases = _load_cases()
        assert len(cases) >= 80, f"Expected >= 80 cases, got {len(cases)}"

    def test_all_required_fields(self) -> None:
        cases = _load_cases()
        for case in cases:
            assert "id" in case, f"Case missing 'id': {case}"
            assert "category" in case, f"Case {case['id']} missing 'category'"
            assert "input" in case, f"Case {case['id']} missing 'input'"
            assert "template" in case, f"Case {case['id']} missing 'template'"
            assert "expected" in case, f"Case {case['id']} missing 'expected'"
            # Must have at least one of the new schema fields
            exp = case["expected"]
            has_new = any(
                k in exp
                for k in ("hard_preserve", "semantic_preserve", "forbidden")
            )
            has_old = any(
                k in exp
                for k in ("must_preserve", "should_not_include")
            )
            assert has_new or has_old, (
                f"Case {case['id']} has no preserve/forbidden fields"
            )

    def test_unique_ids(self) -> None:
        cases = _load_cases()
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids)), f"Duplicate case IDs found: {[i for i in ids if ids.count(i) > 1]}"

    def test_categories_cover_all(self) -> None:
        cases = _load_cases()
        categories = {c["category"] for c in cases}
        expected_categories = {
            "fact_preservation",
            "business_request",
            "refusal",
            "warning",
            "experiment_templates",
            "strong_experiment_templates",
            "safety_boundary",
            "custom_prompt",
            "complex",
        }
        assert categories == expected_categories, f"Missing categories: {expected_categories - categories}"

    def test_semantic_preserve_format(self) -> None:
        """semantic_preserve entries must be str or dict with canonical+allowed."""
        cases = _load_cases()
        for case in cases:
            exp = case.get("expected", {})
            for entry in exp.get("semantic_preserve", []):
                if isinstance(entry, dict):
                    assert "canonical" in entry, (
                        f"Case {case['id']}: semantic_preserve dict missing 'canonical'"
                    )
                    assert "allowed" in entry, (
                        f"Case {case['id']}: semantic_preserve dict missing 'allowed'"
                    )
                    assert isinstance(entry["allowed"], list), (
                        f"Case {case['id']}: semantic_preserve.allowed must be list"
                    )


# ---------------------------------------------------------------------------
# Template reference tests
# ---------------------------------------------------------------------------


class TestTemplateReferences:
    def test_all_templates_in_style_templates(self) -> None:
        cases = _load_cases()
        for case in cases:
            template = case["template"]
            assert template in TEMPLATES, (
                f"Case {case['id']} references unknown template '{template}'"
            )

    def test_experiment_templates_used(self) -> None:
        cases = _load_cases()
        experiment_templates = {c["template"] for c in cases if c["category"] == "experiment_templates"}
        for t in experiment_templates:
            assert t.startswith("実験:"), f"Expected 実験: prefix for experiment template '{t}'"

    def test_strong_experiment_templates_used(self) -> None:
        cases = _load_cases()
        strong_templates = {c["template"] for c in cases if c["category"] == "strong_experiment_templates"}
        for t in strong_templates:
            assert t.startswith("実験強:"), f"Expected 実験強: prefix for strong template '{t}'"

    def test_custom_template_only_in_custom_category(self) -> None:
        cases = _load_cases()
        for case in cases:
            if case["template"] == "カスタム":
                assert case["category"] == "custom_prompt", (
                    f"Case {case['id']} uses カスタム template but category is '{case['category']}'"
                )


# ---------------------------------------------------------------------------
# Expected result validation tests
# ---------------------------------------------------------------------------


class TestExpectedResults:
    def test_preserve_nonempty_for_fact_cases(self) -> None:
        cases = _load_cases()
        for case in cases:
            if case["category"] in ("fact_preservation", "business_request"):
                exp = case["expected"]
                has_preserve = (
                    len(exp.get("hard_preserve", [])) > 0
                    or len(exp.get("must_preserve", [])) > 0
                )
                assert has_preserve, (
                    f"Case {case['id']} ({case['category']}) has empty preserve fields"
                )

    def test_safety_boundary_no_hard_preserve(self) -> None:
        cases = _load_cases()
        for case in cases:
            if case["category"] == "safety_boundary":
                exp = case["expected"]
                assert len(exp.get("hard_preserve", [])) == 0, (
                    f"Safety case {case['id']} has non-empty hard_preserve"
                )
                assert len(exp.get("must_preserve", [])) == 0, (
                    f"Safety case {case['id']} has non-empty must_preserve"
                )

    def test_complex_cases_exist(self) -> None:
        cases = _load_cases()
        complex_cases = [c for c in cases if c["category"] == "complex"]
        assert len(complex_cases) >= 4, f"Expected >= 4 complex cases, got {len(complex_cases)}"


# ---------------------------------------------------------------------------
# File existence tests
# ---------------------------------------------------------------------------


class TestEvalFiles:
    def test_runner_script_exists(self) -> None:
        assert RUNNER_PATH.exists()

    def test_summarizer_script_exists(self) -> None:
        assert SUMMARIZER_PATH.exists()

    def test_judge_prompt_exists(self) -> None:
        assert JUDGE_PROMPT_PATH.exists()

    def test_results_dir_exists(self) -> None:
        results_dir = EVALS_DIR / "results"
        assert results_dir.exists()
        assert (results_dir / ".gitkeep").exists()
