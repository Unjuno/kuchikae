"""Tests for prompt evaluation dataset structure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CASES_FILE = FIXTURES_DIR / "prompt_eval_cases.json"
RUBRIC_FILE = FIXTURES_DIR / "prompt_eval_rubric.json"


def _load_cases() -> list[dict]:
    with open(CASES_FILE, encoding="utf-8") as f:
        return json.load(f)


def _load_rubric() -> dict:
    with open(RUBRIC_FILE, encoding="utf-8") as f:
        return json.load(f)


class TestPromptEvalCasesStructure:
    def test_file_is_valid_json(self) -> None:
        cases = _load_cases()
        assert isinstance(cases, list)

    def test_has_at_least_30_cases(self) -> None:
        cases = _load_cases()
        assert len(cases) >= 30

    def test_ids_are_unique(self) -> None:
        cases = _load_cases()
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids))

    def test_required_fields_present(self) -> None:
        cases = _load_cases()
        required = {"id", "category", "input", "intent", "risk_level", "allowed_freedom"}
        for case in cases:
            assert required.issubset(case.keys()), f"Missing fields in {case['id']}"

    def test_allowed_freedom_values(self) -> None:
        cases = _load_cases()
        valid = {"high", "medium", "low"}
        for case in cases:
            assert case["allowed_freedom"] in valid, f"Invalid freedom in {case['id']}"

    def test_risk_level_values(self) -> None:
        cases = _load_cases()
        valid = {"low", "medium", "high"}
        for case in cases:
            assert case["risk_level"] in valid, f"Invalid risk_level in {case['id']}"

    def test_must_preserve_is_list(self) -> None:
        cases = _load_cases()
        for case in cases:
            assert isinstance(case["must_preserve"], list), f"must_preserve not list in {case['id']}"

    def test_forbidden_is_list(self) -> None:
        cases = _load_cases()
        for case in cases:
            assert isinstance(case["forbidden"], list), f"forbidden not list in {case['id']}"

    def test_not_only_greetings(self) -> None:
        cases = _load_cases()
        categories = [c["category"] for c in cases]
        greeting_count = categories.count("social_greeting")
        assert greeting_count < len(cases) * 0.5, "Too many greetings"

    def test_all_freedom_levels_present(self) -> None:
        cases = _load_cases()
        freedoms = {c["allowed_freedom"] for c in cases}
        assert freedoms == {"high", "medium", "low"}

    def test_must_preserve_cases_exist(self) -> None:
        cases = _load_cases()
        with_preserve = [c for c in cases if c["must_preserve"]]
        assert len(with_preserve) >= 10


class TestPromptEvalRubricStructure:
    def test_file_is_valid_json(self) -> None:
        rubric = _load_rubric()
        assert isinstance(rubric, dict)

    def test_has_required_dimensions(self) -> None:
        rubric = _load_rubric()
        expected = {
            "faithfulness",
            "style_strength",
            "spoken_naturalness",
            "interestingness",
            "conciseness",
            "no_meta_output",
        }
        assert expected.issubset(rubric.keys())

    def test_each_dimension_has_description(self) -> None:
        rubric = _load_rubric()
        for dim, data in rubric.items():
            assert "description" in data, f"Missing description in {dim}"
            assert "criteria" in data, f"Missing criteria in {dim}"
            assert "scoring" in data, f"Missing scoring in {dim}"


class TestPromptEvalCasesContent:
    def test_categories_are_diverse(self) -> None:
        cases = _load_cases()
        categories = {c["category"] for c in cases}
        assert len(categories) >= 10

    def test_sample_inputs_are_strings(self) -> None:
        cases = _load_cases()
        for case in cases:
            assert isinstance(case["input"], str)
            assert len(case["input"]) > 0
