"""Tests for kuchikae.models (model setup/repair)."""

from __future__ import annotations

from unittest.mock import patch

from kuchikae.models import (
    ModelSpec,
    ModelStatus,
    SetupReport,
    _model_specs,
    check_model,
    check_models,
    setup_model,
    setup_models,
    repair_models,
    print_model_status,
    print_setup_report,
)


def test_model_specs_returns_list() -> None:
    specs = _model_specs()
    assert isinstance(specs, list)
    assert len(specs) >= 3
    categories = {s.category for s in specs}
    assert "stt" in categories
    assert "tts" in categories


def test_model_specs_has_required_fields() -> None:
    for spec in _model_specs():
        assert spec.name
        assert spec.category in ("stt", "tts", "emotion")
        assert spec.model_type in ("whisper", "hf_repo", "hf_file")
        assert spec.default_id
        assert spec.description


def test_check_model_returns_model_status() -> None:
    specs = _model_specs()
    for spec in specs:
        status = check_model(spec)
        assert isinstance(status, ModelStatus)
        assert status.name == spec.name
        assert status.category == spec.category
        assert status.status in ("ok", "missing", "error", "fixable")


def test_check_models_with_category_filter() -> None:
    stt_statuses = check_models(category="stt")
    assert all(s.category == "stt" for s in stt_statuses)
    assert len(stt_statuses) >= 1


def test_check_models_without_filter() -> None:
    statuses = check_models()
    assert len(statuses) >= 3


@patch("kuchikae.models._check_whisper_model")
def test_check_model_whisper_type(mock_check) -> None:
    mock_check.return_value = ModelStatus(
        name="test", category="stt", status="ok",
    )
    spec = ModelSpec(
        name="test", category="stt", model_type="whisper",
        default_id="small",
    )
    result = check_model(spec)
    assert result.status == "ok"
    mock_check.assert_called_once_with(spec)


@patch("kuchikae.models._setup_whisper_model")
def test_setup_model_delegates_to_type_handler(mock_setup) -> None:
    mock_setup.return_value = ModelStatus(
        name="test", category="stt", status="ok",
    )
    spec = ModelSpec(
        name="test", category="stt", model_type="whisper",
        default_id="small",
    )
    result = setup_model(spec)
    assert result.status == "ok"
    mock_setup.assert_called_once_with(spec)


@patch("kuchikae.models._setup_whisper_model")
def test_setup_model_repair_flag(mock_setup) -> None:
    mock_setup.return_value = ModelStatus(
        name="test", category="stt", status="ok",
    )
    spec = ModelSpec(
        name="test", category="stt", model_type="whisper",
        default_id="small",
    )
    setup_model(spec, repair=True)
    mock_setup.assert_called_once_with(spec)


@patch("kuchikae.models.setup_model")
def test_setup_models_aggregates_results(mock_setup_model) -> None:
    mock_setup_model.return_value = ModelStatus(
        name="test", category="stt", status="ok",
    )
    report = setup_models(category="stt")
    assert isinstance(report, SetupReport)
    assert len(report.models) >= 1
    assert report.errors == []


@patch("kuchikae.models.setup_model")
def test_setup_models_collects_errors(mock_setup_model) -> None:
    mock_setup_model.return_value = ModelStatus(
        name="failing-model", category="stt", status="error",
        error="download failed",
    )
    report = setup_models(category="stt")
    assert len(report.errors) == 1
    assert "failing-model" in report.errors[0]


@patch("kuchikae.models.setup_models")
def test_repair_models_calls_setup_with_repair(mock_setup) -> None:
    mock_setup.return_value = SetupReport()
    repair_models(category="stt")
    mock_setup.assert_called_once_with(category="stt", repair=True)


def test_print_model_status_ok(capsys) -> None:
    status = ModelStatus(
        name="test-model", category="stt", status="ok",
        path="/some/path",
    )
    print_model_status(status)
    captured = capsys.readouterr()
    assert "[OK]" in captured.out
    assert "test-model" in captured.out
    assert "/some/path" in captured.out


def test_print_model_status_missing(capsys) -> None:
    status = ModelStatus(
        name="test-model", category="stt", status="missing",
        error="not found",
    )
    print_model_status(status)
    captured = capsys.readouterr()
    assert "[MISSING]" in captured.out
    assert "not found" in captured.out


def test_print_model_status_error(capsys) -> None:
    status = ModelStatus(
        name="test-model", category="stt", status="error",
        error="import failed",
    )
    print_model_status(status)
    captured = capsys.readouterr()
    assert "[ERROR]" in captured.out
    assert "import failed" in captured.out


def test_print_setup_report(capsys) -> None:
    report = SetupReport(
        models=[
            ModelStatus(name="m1", category="stt", status="ok"),
            ModelStatus(name="m2", category="tts", status="missing", error="not found"),
        ],
        errors=["m2: not found"],
    )
    print_setup_report(report, title="Test Report")
    captured = capsys.readouterr()
    assert "Test Report" in captured.out
    assert "[OK] m1" in captured.out
    assert "[MISSING] m2" in captured.out
    assert "m2: not found" in captured.out


def test_setup_report_defaults() -> None:
    report = SetupReport()
    assert report.models == []
    assert report.errors == []


def test_model_status_defaults() -> None:
    status = ModelStatus(name="x", category="stt", status="ok")
    assert status.path is None
    assert status.error is None
    assert status.repairable is False


def test_check_whisper_missing_cache(tmp_path) -> None:
    spec = ModelSpec(
        name="faster-whisper", category="stt", model_type="whisper",
        default_id="small", env_var="WHISPER_MODEL_SIZE",
    )
    with patch("kuchikae.models.Path.home", return_value=tmp_path):
        status = check_model(spec)
        assert status.status in ("missing", "error")
        assert status.repairable is True


def test_check_hf_file_missing(tmp_path) -> None:
    spec = ModelSpec(
        name="irodori-tts", category="tts", model_type="hf_file",
        default_id="Aratako/Irodori-TTS-500M-v3",
    )
    with patch("kuchikae.models.Path.home", return_value=tmp_path):
        status = check_model(spec)
        assert status.status in ("missing", "error")
        assert status.repairable is True


def test_check_hf_repo_missing(tmp_path) -> None:
    spec = ModelSpec(
        name="irodori-codec", category="tts", model_type="hf_repo",
        default_id="Aratako/Semantic-DACVAE-Japanese-32dim",
    )
    with patch("kuchikae.models.Path.home", return_value=tmp_path):
        status = check_model(spec)
        assert status.status in ("missing", "error")
        assert status.repairable is True
