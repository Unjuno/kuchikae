"""Tests for the text transform evaluation dataset and scripts."""

from __future__ import annotations

from pathlib import Path

import yaml

from kuchikae.ui.templates import TEMPLATES


EVALS_DIR = Path(__file__).parent.parent / "evals"
CASES_PATH = EVALS_DIR / "text_transform_cases.yaml"
RUNNER_PATH = EVALS_DIR / "run_text_transform_eval.py"
SUMMARIZER_PATH = EVALS_DIR / "summarize_eval.py"
JUDGE_PROMPT_PATH = EVALS_DIR / "judge_prompt.md"


# ---------------------------------------------------------------------------
# Import judge logic for unit tests
# ---------------------------------------------------------------------------

import sys

sys.path.insert(0, str(EVALS_DIR))
from run_text_transform_eval import (
    EvalCase,
    ExpectedResult,
    SemanticPreserveEntry,
    judge_rule,
)


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


# ---------------------------------------------------------------------------
# Judge logic unit tests
# ---------------------------------------------------------------------------


def _make_case(
    case_id: str = "test-001",
    input_text: str = "テスト入力です",
    hard: list[str] | None = None,
    semantic: list[dict] | None = None,
    forbidden: list[str] | None = None,
) -> EvalCase:
    exp = ExpectedResult(
        hard_preserve=hard or [],
        semantic_preserve=[SemanticPreserveEntry.from_raw(s) for s in (semantic or [])],
        forbidden=forbidden or [],
    )
    return EvalCase(
        id=case_id,
        category="test",
        input=input_text,
        template="自然に",
        expected=exp,
    )


class TestJudgeRule:
    def test_semantic_variant_hit_is_pass(self) -> None:
        case = _make_case(
            semantic=[{"canonical": "クライアント", "allowed": ["クライアント", "お客様", "顧客"]}],
        )
        result = judge_rule(case, "お客様にお知らせください", "prompted_rule")
        assert result.verdict == "pass"
        assert result.semantic_preserve_hits.get("クライアント") is True

    def test_semantic_missing_is_warn(self) -> None:
        case = _make_case(
            semantic=[{"canonical": "クライアント", "allowed": ["クライアント", "お客様"]}],
        )
        result = judge_rule(case, "田中さんにお知らせください", "prompted_rule")
        assert result.verdict == "warn"
        assert result.failure_type == "semantic_loss"

    def test_hard_missing_is_fail(self) -> None:
        case = _make_case(hard=["田中さん"])
        result = judge_rule(case, "佐藤さんにお知らせください", "prompted_rule")
        assert result.verdict == "fail"
        assert result.failure_type == "meaning_loss"

    def test_forbidden_hit_is_fail_even_prompted_rule(self) -> None:
        case = _make_case(forbidden=["パスワード"])
        result = judge_rule(case, "パスワードを教えてください", "prompted_rule")
        assert result.verdict == "fail"
        assert result.failure_type == "unsafe"

    def test_cot_leak_thought_tag(self) -> None:
        case = _make_case()
        result = judge_rule(case, "<think>よろしく</think> 今日はいい天気ですね", "prompted_rule")
        assert result.verdict == "fail"
        assert result.failure_type == "cot_leak"

    def test_cot_leak_thought_tag_case_insensitive(self) -> None:
        case = _make_case()
        result = judge_rule(case, "<THOUGHT>よろしく</THOUGHT> 今日はいい天気ですね", "prompted_rule")
        assert result.verdict == "fail"
        assert result.failure_type == "cot_leak"

    def test_cot_leak_thought_only(self) -> None:
        case = _make_case()
        result = judge_rule(case, "考え中<thought> 計画を立て中", "prompted_rule")
        assert result.verdict == "fail"
        assert result.failure_type == "cot_leak"

    def test_cot_leak_closing_thought(self) -> None:
        case = _make_case()
        result = judge_rule(case, "今日は</thought>いい天気", "prompted_rule")
        assert result.verdict == "fail"
        assert result.failure_type == "cot_leak"

    def test_empty_output_is_fail(self) -> None:
        case = _make_case()
        result = judge_rule(case, "", "prompted_rule")
        assert result.verdict == "fail"
        assert result.failure_type == "empty_output"

    def test_template_leak_is_fail(self) -> None:
        case = _make_case()
        result = judge_rule(case, "[STYLE_TEMPLATE: 自然に] 今日はいい天気", "prompted_rule")
        assert result.verdict == "fail"
        assert result.failure_type == "template_leak"

    def test_echo_is_fail(self) -> None:
        case = _make_case(input_text="今日はいい天気ですね")
        result = judge_rule(case, "今日はいい天気ですね", "prompted_rule")
        assert result.verdict == "fail"
        assert result.failure_type == "style_weak"

    def test_pass_when_all_good(self) -> None:
        case = _make_case(
            hard=["田中さん"],
            semantic=[{"canonical": "確認", "allowed": ["確認", "チェック"]}],
        )
        result = judge_rule(case, "田中さん、確認してください", "prompted_rule")
        assert result.verdict == "pass"

    def test_hard_pass_semantic_warn(self) -> None:
        case = _make_case(
            hard=["田中さん"],
            semantic=[{"canonical": "確認", "allowed": ["確認", "チェック"]}],
        )
        result = judge_rule(case, "田中さん、お知らせください", "prompted_rule")
        assert result.verdict == "warn"
        assert result.failure_type == "semantic_loss"
