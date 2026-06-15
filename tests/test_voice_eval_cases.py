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

    def test_dry_run_tts_only_produces_skip_verdicts(self):
        from evals.run_voice_eval import process_case

        cases = load_cases()
        for case in cases:
            result = process_case(case, pipeline=None, backend="irodori", mode="tts-only", dry_run=True)
            assert isinstance(result, VoiceEvalResult)
            assert result.verdict == "skip"
            assert result.case_id == case.id

    def test_dry_run_pipeline_produces_skip_verdicts(self):
        from evals.run_voice_eval import process_case

        cases = load_cases()
        for case in cases:
            result = process_case(case, pipeline=None, backend="irodori", mode="pipeline", dry_run=True)
            assert isinstance(result, VoiceEvalResult)
            assert result.verdict == "skip"
            assert result.case_id == case.id

    def test_dry_run_result_has_mode_field(self):
        from evals.run_voice_eval import process_case

        cases = load_cases()
        for case in cases:
            result = process_case(case, pipeline=None, backend="irodori", mode="tts-only", dry_run=True)
            assert result.mode == "tts-only"
            result2 = process_case(case, pipeline=None, backend="irodori", mode="pipeline", dry_run=True)
            assert result2.mode == "pipeline"

    def test_dry_run_write_and_read_jsonl_has_mode(self, tmp_path):
        cases = load_cases()
        from evals.run_voice_eval import process_case

        results = [process_case(c, pipeline=None, backend="irodori", mode="tts-only", dry_run=True) for c in cases]
        output_path = tmp_path / "voice_results.jsonl"
        write_results(results, output_path)
        assert output_path.exists()

        from evals.summarize_voice_eval import load_results

        loaded = load_results(output_path)
        assert len(loaded) == len(results)
        for i, row in enumerate(loaded):
            assert row["case_id"] == results[i].case_id
            assert row["verdict"] == "skip"
            assert row["mode"] == "tts-only"

    def test_dry_run_with_openvoice_backend(self):
        """--backend openvoice should not silently use Irodori."""
        from evals.run_voice_eval import process_case, load_cases

        cases = load_cases()
        for case in cases[:2]:
            result = process_case(case, pipeline=None, backend="openvoice", mode="tts-only", dry_run=True)
            assert result.voice_backend == "openvoice", f"Expected openvoice, got {result.voice_backend}"


# ---------------------------------------------------------------------------
# Voice backend factory
# ---------------------------------------------------------------------------


class TestBuildVoiceBackend:
    def test_build_irodori(self):
        from evals.run_voice_eval import _build_voice_backend
        backend = _build_voice_backend("irodori")
        from kuchikae.backends.voice_output import IrodoriTTSVoiceOutputBackend
        assert isinstance(backend, IrodoriTTSVoiceOutputBackend)

    def test_build_openvoice(self):
        from evals.run_voice_eval import _build_voice_backend
        backend = _build_voice_backend("openvoice")
        from kuchikae.backends.voice_output import OpenVoiceOutputBackend
        assert isinstance(backend, OpenVoiceOutputBackend)

    def test_unsupported_backend_raises(self):
        from evals.run_voice_eval import _build_voice_backend
        for name in ("f5-tts", "cosyvoice", "indextts", "rvc", "xtts"):
            with pytest.raises(NotImplementedError, match="Unsupported eval backend"):
                _build_voice_backend(name)

    def test_openvoice_backend_in_process_case(self):
        """process_case with backend=openvoice should set voice_backend field correctly."""
        from evals.run_voice_eval import process_case, load_cases
        cases = load_cases()
        for case in cases[:3]:
            result = process_case(case, pipeline=None, backend="openvoice", mode="tts-only", dry_run=True)
            assert result.voice_backend == "openvoice"


# ---------------------------------------------------------------------------
# Duration ratio (sample rate aware)
# ---------------------------------------------------------------------------


class TestDurationRatio:
    def test_sample_rate_aware_16000(self, tmp_path):
        """duration_ratio must use the actual sample rate returned by sf.read."""
        import numpy as np
        import soundfile as sf
        from evals.run_voice_eval import _compute_duration_ratio

        sr = 16000
        dur = 2.0
        samples = np.zeros(int(dur * sr), dtype=np.float32)
        in_path = str(tmp_path / "in_16k.wav")
        sf.write(in_path, samples, sr)

        out_sr = 44100
        out_dur = 1.0
        out_samples = np.zeros(int(out_dur * out_sr), dtype=np.float32)
        out_path = str(tmp_path / "out_44k.wav")
        sf.write(out_path, out_samples, out_sr)

        ratio = _compute_duration_ratio(in_path, out_path)
        assert ratio is not None
        assert ratio == pytest.approx(0.5, abs=0.01), f"Expected 0.5, got {ratio}"

    def test_sample_rate_aware_equal_duration(self, tmp_path):
        import numpy as np
        import soundfile as sf
        from evals.run_voice_eval import _compute_duration_ratio

        sr = 48000
        dur = 1.5
        samples = np.zeros(int(dur * sr), dtype=np.float32)
        in_path = str(tmp_path / "in_48k.wav")
        out_path = str(tmp_path / "out_48k.wav")
        sf.write(in_path, samples, sr)
        sf.write(out_path, samples, sr)

        ratio = _compute_duration_ratio(in_path, out_path)
        assert ratio is not None
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_sample_rate_aware_zero_input(self, tmp_path):
        import numpy as np
        import soundfile as sf
        from evals.run_voice_eval import _compute_duration_ratio

        in_path = str(tmp_path / "empty.wav")
        samples = np.array([], dtype=np.float32)
        sf.write(in_path, samples, 16000)
        out_path = str(tmp_path / "out.wav")
        sf.write(out_path, samples, 16000)

        ratio = _compute_duration_ratio(in_path, out_path)
        assert ratio is None

    def test_sample_rate_aware_missing_file(self):
        from evals.run_voice_eval import _compute_duration_ratio
        ratio = _compute_duration_ratio("/nonexistent/input.wav", "/nonexistent/output.wav")
        assert ratio is None

    def test_duration_ratio_warn_high_has_reason(self, tmp_path):
        """Warn verdict for duration_ratio > 4.0 should include failure_reason."""
        import numpy as np
        import soundfile as sf
        from evals.run_voice_eval import _compute_duration_ratio

        in_sr, in_dur = 44100, 2.0
        sf.write(str(tmp_path / "in.wav"), np.zeros(int(in_dur * in_sr)), in_sr)
        out_sr, out_dur = 8000, 12.0
        sf.write(str(tmp_path / "out.wav"), np.zeros(int(out_dur * out_sr)), out_sr)

        ratio = _compute_duration_ratio(str(tmp_path / "in.wav"), str(tmp_path / "out.wav"))
        assert ratio is not None
        assert ratio > 4.0, f"Expected ratio > 4.0, got {ratio}"

    def test_duration_ratio_warn_low_has_reason(self, tmp_path):
        """Warn verdict for duration_ratio < 0.3 should include failure_reason."""
        import numpy as np
        import soundfile as sf
        from evals.run_voice_eval import _compute_duration_ratio

        in_sr, in_dur = 8000, 10.0
        sf.write(str(tmp_path / "in.wav"), np.zeros(int(in_dur * in_sr)), in_sr)
        out_sr, out_dur = 44100, 1.0
        sf.write(str(tmp_path / "out.wav"), np.zeros(int(out_dur * out_sr)), out_sr)

        ratio = _compute_duration_ratio(str(tmp_path / "in.wav"), str(tmp_path / "out.wav"))
        assert ratio is not None
        assert ratio < 0.3, f"Expected ratio < 0.3, got {ratio}"


# ---------------------------------------------------------------------------
# Mode validation
# ---------------------------------------------------------------------------


class TestModeValidation:
    def test_transformed_text_rejects_dummy_stt(self, tmp_path):
        """Verify that [DUMMY_STT_OUTPUT] in transformed_text is caught."""
        from evals.run_voice_eval import VoiceEvalResult

        result = VoiceEvalResult(
            case_id="test-001",
            input_audio="test.wav",
            output_audio=None,
            template=u"自然に",
            input_text="test",
            transformed_text="[DUMMY_STT_OUTPUT] something",
            voice_backend="irodori",
            mode="tts-only",
            verdict="fail",
            failure_reason="transformed_text contains [DUMMY_STT_OUTPUT] (STT leak in tts-only mode)",
        )
        assert result.verdict == "fail"
        assert "DUMMY_STT_OUTPUT" in result.failure_reason

    def test_missing_fixture_returns_skip_in_both_modes(self, tmp_path):
        from evals.run_voice_eval import process_case, VoiceEvalCase

        case = VoiceEvalCase(
            id="missing-fixture",
            input_audio="nonexistent.wav",
            input_text="test",
            template=u"自然に",
        )
        for mode in ("tts-only", "pipeline"):
            result = process_case(case, pipeline=None, backend="irodori", mode=mode, dry_run=False)
            assert result.verdict == "skip"
            assert "fixture not found" in result.failure_reason

    def test_cli_defaults_to_tts_only(self):
        from evals.run_voice_eval import parse_args
        args = parse_args([])
        assert args.mode == "tts-only"
        args2 = parse_args(["--mode", "pipeline"])
        assert args2.mode == "pipeline"

    def test_cli_backend_defaults_to_irodori(self):
        from evals.run_voice_eval import parse_args
        args = parse_args([])
        assert args.backend == "irodori"

    def test_cli_backend_choices_limited(self):
        """CLI should reject unsupported backends."""
        from evals.run_voice_eval import parse_args
        for name in ("f5-tts", "cosyvoice", "indextts", "rvc", "xtts"):
            with pytest.raises(SystemExit):
                parse_args(["--backend", name])


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
            {"case_id": "ns-001", "verdict": "skip", "voice_backend": "irodori", "template": u"自然に",
             "speaker_similarity": None, "duration_ratio": None, "failure_reason": "dry-run",
             "mode": "tts-only"},
            {"case_id": "ns-002", "verdict": "skip", "voice_backend": "irodori", "template": u"自然に",
             "speaker_similarity": None, "duration_ratio": None, "failure_reason": "dry-run",
             "mode": "tts-only"},
        ]
        from evals.summarize_voice_eval import compute_summary

        summary = compute_summary(results)
        assert summary["overall"]["total"] == 2
        assert summary["overall"]["skipped"] == 2
        assert summary["average_speaker_similarity"] is None

    def test_summary_has_by_mode(self):
        from evals.summarize_voice_eval import compute_summary

        results = [
            {"case_id": "a", "verdict": "pass", "voice_backend": "irodori", "template": u"自然に",
             "speaker_similarity": None, "duration_ratio": None, "failure_reason": "", "mode": "tts-only"},
            {"case_id": "b", "verdict": "fail", "voice_backend": "openvoice", "template": u"自然に",
             "speaker_similarity": None, "duration_ratio": None, "failure_reason": "err", "mode": "pipeline"},
        ]
        summary = compute_summary(results)
        assert "by_mode" in summary
        assert summary["by_mode"]["tts-only"]["total"] == 1
        assert summary["by_mode"]["tts-only"]["passed"] == 1
        assert summary["by_mode"]["pipeline"]["total"] == 1
        assert summary["by_mode"]["pipeline"]["failed"] == 1

    def test_summary_by_mode_in_text_output(self, tmp_path):
        """by_mode should appear in text format output."""
        from evals.summarize_voice_eval import compute_summary, format_text

        results = [
            {"case_id": "a", "verdict": "pass", "voice_backend": "irodori", "template": u"自然に",
             "speaker_similarity": None, "duration_ratio": None, "failure_reason": "", "mode": "tts-only"},
        ]
        summary = compute_summary(results)
        text = format_text(summary)
        assert "by mode" in text.lower()
        assert "tts-only" in text

    def test_summary_by_mode_in_markdown_output(self, tmp_path):
        """by_mode should appear in markdown format output."""
        from evals.summarize_voice_eval import compute_summary, format_markdown

        results = [
            {"case_id": "a", "verdict": "pass", "voice_backend": "irodori", "template": u"自然に",
             "speaker_similarity": None, "duration_ratio": None, "failure_reason": "", "mode": "tts-only"},
        ]
        summary = compute_summary(results)
        md = format_markdown(summary)
        assert "By mode" in md
        assert "tts-only" in md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestVoiceEvalCLI:
    def test_dry_run_cli_default_mode(self):
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
        for line in lines:
            record = json.loads(line)
            assert "mode" in record, f"Missing mode field in {record['case_id']}"
            assert record["mode"] == "tts-only", f"Expected tts-only, got {record['mode']}"
        output_path.unlink(missing_ok=True)

    def test_dry_run_cli_pipeline_mode(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "evals/run_voice_eval.py"), "--dry-run",
             "--mode", "pipeline",
             "--out", str(RESULTS_DIR / "test_voice_cli_pipeline.jsonl")],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output_path = RESULTS_DIR / "test_voice_cli_pipeline.jsonl"
        assert output_path.exists()
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) >= 8
        for line in lines:
            record = json.loads(line)
            assert record["mode"] == "pipeline", f"Expected pipeline, got {record['mode']}"
        output_path.unlink(missing_ok=True)

    def test_summarize_cli_empty_file(self, tmp_path):
        jsonl = tmp_path / "empty.jsonl"
        jsonl.write_text("")
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "evals/summarize_voice_eval.py"), str(jsonl)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0

    def test_dry_run_cli_openvoice_backend(self):
        """--backend openvoice should not silently use Irodori."""
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "evals/run_voice_eval.py"), "--dry-run",
             "--backend", "openvoice",
             "--out", str(RESULTS_DIR / "test_voice_cli_openvoice.jsonl")],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output_path = RESULTS_DIR / "test_voice_cli_openvoice.jsonl"
        assert output_path.exists()
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) >= 8
        for line in lines:
            record = json.loads(line)
            assert record["voice_backend"] == "openvoice", f"Expected openvoice, got {record['voice_backend']}"
        output_path.unlink(missing_ok=True)

    def test_summarize_cli_shows_by_mode(self, tmp_path):
        """Verify summarize CLI output shows by_mode for real results file."""
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "evals/summarize_voice_eval.py"),
             str(RESULTS_DIR / "voice_eval_irodori_tts_only_baseline.jsonl")],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        assert "by mode" in result.stdout.lower()
