"""Tests for voice eval cases structure and runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from evals.run_voice_eval import (
    CASES_PATH,
    FIXTURES_DIR,
    RESULTS_DIR,
    VoiceEvalResult,
    load_cases,
    validate_fixtures,
    write_results,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# YAML structure
# ---------------------------------------------------------------------------


class TestVoiceEvalCasesStructure:
    def test_yaml_exists(self):
        assert CASES_PATH.exists(), f"voice_cases.yaml not found at {CASES_PATH}"

    def test_yaml_is_valid(self):
        with open(CASES_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None, "YAML is empty"
        assert "cases" in data, "YAML missing 'cases' key"

    def test_cases_are_list(self):
        with open(CASES_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data["cases"], list)

    def test_minimum_case_count(self):
        cases = load_cases()
        assert len(cases) >= 8, f"Expected at least 8 cases, got {len(cases)}"

    def test_all_required_fields(self):
        cases = load_cases()
        for case in cases:
            assert case.id, f"Case missing id: {case}"
            assert case.input_audio, f"Case {case.id} missing input_audio"
            assert case.input_text, f"Case {case.id} missing input_text"
            assert case.template, f"Case {case.id} missing template"

    def test_unique_ids(self):
        cases = load_cases()
        ids = [c.id for c in cases]
        duplicates = [i for i in ids if ids.count(i) > 1]
        assert not duplicates, f"Duplicate case IDs: {set(duplicates)}"

    def test_expected_fields(self):
        cases = load_cases()
        for case in cases:
            assert case.expected.emotion in (
                "happy", "anger", "sad", "calm", "neutral"
            ), f"Case {case.id}: unexpected emotion '{case.expected.emotion}'"
            assert case.expected.voice_style in (
                "auto", "natural", "calm", "bright", "slow_clear"
            ), f"Case {case.id}: unexpected voice_style '{case.expected.voice_style}'"

    def test_should_preserve_and_not_preserve_are_lists(self):
        cases = load_cases()
        for case in cases:
            assert isinstance(case.expected.should_preserve, list), f"Case {case.id}: should_preserve not a list"
            assert isinstance(case.expected.should_not_preserve, list), f"Case {case.id}: should_not_preserve not a list"


# ---------------------------------------------------------------------------
# Fixture validation
# ---------------------------------------------------------------------------


class TestVoiceEvalFixtures:
    def test_fixtures_dir_exists(self):
        assert FIXTURES_DIR.exists(), f"Fixtures dir not found: {FIXTURES_DIR}"

    def test_gitkeep_exists(self):
        assert (FIXTURES_DIR / ".gitkeep").exists()

    def test_missing_fixtures_detected(self):
        """Cases referencing non-existent fixtures should be reported."""
        cases = load_cases()
        errors = validate_fixtures(cases)
        # All fixtures are expected to be missing in the initial skeleton
        # (fixtures will be added later)
        for case in cases:
            fixture = FIXTURES_DIR / case.input_audio
            if not fixture.exists():
                assert case.id in errors, f"Case {case.id}: expected fixture error for {fixture}"
            else:
                assert case.id not in errors, f"Case {case.id}: fixture exists but error reported"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestVoiceEvalRunner:
    def test_load_cases_returns_list(self):
        cases = load_cases()
        assert isinstance(cases, list)
        assert len(cases) > 0

    def test_dry_run_produces_skip_verdicts(self):
        from evals.run_voice_eval import process_case

        cases = load_cases()
        for case in cases:
            result = process_case(case, pipeline=None, backend="irodori", dry_run=True)
            assert isinstance(result, VoiceEvalResult)
            assert result.verdict == "skip"
            assert result.case_id == case.id

    def test_write_and_read_jsonl(self, tmp_path):
        cases = load_cases()
        from evals.run_voice_eval import process_case

        results = [process_case(c, pipeline=None, backend="irodori", dry_run=True) for c in cases]
        output_path = tmp_path / "voice_results.jsonl"
        write_results(results, output_path)
        assert output_path.exists()

        # Read back and verify
        from evals.summarize_voice_eval import load_results

        loaded = load_results(output_path)
        assert len(loaded) == len(results)
        for i, row in enumerate(loaded):
            assert row["case_id"] == results[i].case_id
            assert row["verdict"] == "skip"


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


class TestVoiceEvalSummarizer:
    def test_summarize_empty_results(self):
        from evals.summarize_voice_eval import compute_summary

        summary = compute_summary([])
        assert summary["overall"]["total"] == 0
        assert summary["overall"]["passed"] == 0

    def test_summarize_dry_run_results(self, tmp_path):
        results = [
            {"case_id": "ns-001", "verdict": "skip", "voice_backend": "irodori", "template": "自然に",
             "speaker_similarity": None, "duration_ratio": None, "failure_reason": "dry-run"},
            {"case_id": "ns-002", "verdict": "skip", "voice_backend": "irodori", "template": "自然に",
             "speaker_similarity": None, "duration_ratio": None, "failure_reason": "dry-run"},
        ]
        from evals.summarize_voice_eval import compute_summary

        summary = compute_summary(results)
        assert summary["overall"]["total"] == 2
        assert summary["overall"]["skipped"] == 2
        assert summary["average_speaker_similarity"] is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestVoiceEvalCLI:
    def test_dry_run_cli(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "evals/run_voice_eval.py"), "--dry-run",
             "--out", str(RESULTS_DIR / "test_voice_cli.jsonl")],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output_path = RESULTS_DIR / "test_voice_cli.jsonl"
        assert output_path.exists()
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) >= 8
        # cleanup
        output_path.unlink(missing_ok=True)

    def test_summarize_cli_empty_file(self, tmp_path):
        jsonl = tmp_path / "empty.jsonl"
        jsonl.write_text("")
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "evals/summarize_voice_eval.py"), str(jsonl)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0
