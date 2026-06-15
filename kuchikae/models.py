"""Model setup, check, and repair utilities.

Provides CLI-facing functions for downloading and verifying model weights
used by Kuchikae's real backends (STT, TTS, audio emotion).

All functions return structured ModelStatus objects instead of raising
RuntimeErrors, so the CLI can display results uniformly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    category: str  # "stt" | "tts" | "emotion"
    model_type: str  # "whisper" | "hf_repo" | "hf_file"
    default_id: str
    env_var: str | None = None
    required: bool = True
    description: str = ""


@dataclass
class ModelStatus:
    name: str
    category: str
    status: str  # "ok" | "missing" | "error" | "fixable"
    path: str | None = None
    error: str | None = None
    repairable: bool = False


@dataclass
class SetupReport:
    models: list[ModelStatus] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _default_stt_model_size() -> str:
    return os.environ.get("WHISPER_MODEL_SIZE", "small")


def _default_stt_model_id() -> str:
    return _default_stt_model_size()


def _default_hf_emotion_model() -> str:
    return os.environ.get(
        "KUCHIKAE_AUDIO_EMOTION_MODEL",
        "superb/wav2vec2-base-superb-er",
    )


def _default_irodori_checkpoint() -> str:
    return os.environ.get("IRODORI_MODEL_ID", "Aratako/Irodori-TTS-500M-v3")


def _default_irodori_codec() -> str:
    return os.environ.get("IRODORI_CODEC_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim")


def _model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="faster-whisper",
            category="stt",
            model_type="whisper",
            default_id=_default_stt_model_id(),
            env_var="WHISPER_MODEL_SIZE",
            required=True,
            description=f"Speech-to-text model (default: {_default_stt_model_size()})",
        ),
        ModelSpec(
            name="irodori-tts",
            category="tts",
            model_type="hf_file",
            default_id=_default_irodori_checkpoint(),
            env_var="IRODORI_MODEL_ID",
            required=True,
            description="Text-to-speech checkpoint (Irodori-TTS)",
        ),
        ModelSpec(
            name="irodori-codec",
            category="tts",
            model_type="hf_repo",
            default_id=_default_irodori_codec(),
            env_var="IRODORI_CODEC_REPO",
            required=True,
            description="TTS audio codec (Semantic-DACVAE)",
        ),
        ModelSpec(
            name="audio-emotion",
            category="emotion",
            model_type="hf_repo",
            default_id=_default_hf_emotion_model(),
            env_var="KUCHIKAE_AUDIO_EMOTION_MODEL",
            required=False,
            description="Audio emotion detection model (optional)",
        ),
    ]


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def _check_whisper_model(spec: ModelSpec) -> ModelStatus:
    model_id = os.environ.get(spec.env_var or "", spec.default_id)
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore[import-untyped]
        cache_dir = os.path.join(
            Path.home(), ".cache", "huggingface", "hub",
            f"models--Systran--faster-whisper-{model_id}",
        )
        if os.path.isdir(cache_dir):
            return ModelStatus(
                name=spec.name, category=spec.category, status="ok",
                path=cache_dir,
            )
        return ModelStatus(
            name=spec.name, category=spec.category, status="missing",
            error=f"Cache dir not found: {cache_dir}",
            repairable=True,
        )
    except ImportError:
        return ModelStatus(
            name=spec.name, category=spec.category, status="error",
            error="huggingface_hub not installed",
        )
    except Exception as e:
        return ModelStatus(
            name=spec.name, category=spec.category, status="error",
            error=str(e),
        )


def _check_hf_file_model(spec: ModelSpec) -> ModelStatus:
    repo_id = os.environ.get(spec.env_var or "", spec.default_id)
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore[import-untyped]
        filename = "model.safetensors"
        cache_path = try_to_load_from_cache(
            repo_id=repo_id,
            filename=filename,
        )
        if isinstance(cache_path, Path) and cache_path.exists():
            return ModelStatus(
                name=spec.name, category=spec.category, status="ok",
                path=str(cache_path),
            )
        return ModelStatus(
            name=spec.name, category=spec.category, status="missing",
            error=f"{filename} not cached for {repo_id}",
            repairable=True,
        )
    except ImportError:
        return ModelStatus(
            name=spec.name, category=spec.category, status="error",
            error="huggingface_hub not installed",
        )
    except Exception as e:
        return ModelStatus(
            name=spec.name, category=spec.category, status="error",
            error=str(e),
        )


def _check_hf_repo_model(spec: ModelSpec) -> ModelStatus:
    repo_id = os.environ.get(spec.env_var or "", spec.default_id)
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore[import-untyped]
        # Check for any cached snapshot of this repo
        repo_cache = os.path.join(
            Path.home(), ".cache", "huggingface", "hub",
            f"models--{repo_id.replace('/', '--')}",
        )
        snapshots_dir = os.path.join(repo_cache, "snapshots")
        if os.path.isdir(snapshots_dir) and os.listdir(snapshots_dir):
            return ModelStatus(
                name=spec.name, category=spec.category, status="ok",
                path=snapshots_dir,
            )
        return ModelStatus(
            name=spec.name, category=spec.category, status="missing",
            error=f"No snapshots cached for {repo_id}",
            repairable=True,
        )
    except ImportError:
        return ModelStatus(
            name=spec.name, category=spec.category, status="error",
            error="huggingface_hub not installed",
        )
    except Exception as e:
        return ModelStatus(
            name=spec.name, category=spec.category, status="error",
            error=str(e),
        )


def check_model(spec: ModelSpec) -> ModelStatus:
    if spec.model_type == "whisper":
        return _check_whisper_model(spec)
    elif spec.model_type == "hf_file":
        return _check_hf_file_model(spec)
    elif spec.model_type == "hf_repo":
        return _check_hf_repo_model(spec)
    return ModelStatus(
        name=spec.name, category=spec.category, status="error",
        error=f"Unknown model type: {spec.model_type}",
    )


def check_models(category: str | None = None) -> list[ModelStatus]:
    specs = _model_specs()
    if category:
        specs = [s for s in specs if s.category == category]
    return [check_model(s) for s in specs]


# ---------------------------------------------------------------------------
# Setup (download)
# ---------------------------------------------------------------------------

def _setup_whisper_model(spec: ModelSpec) -> ModelStatus:
    model_id = os.environ.get(spec.env_var or "", spec.default_id)
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        logger.info("Downloading whisper model '%s'...", model_id)
        model = WhisperModel(model_id, device="cpu", compute_type="int8")
        del model
        logger.info("Whisper model '%s' ready.", model_id)
        return ModelStatus(
            name=spec.name, category=spec.category, status="ok",
        )
    except ImportError:
        return ModelStatus(
            name=spec.name, category=spec.category, status="error",
            error="faster-whisper not installed. Run: uv sync --extra real",
        )
    except Exception as e:
        return ModelStatus(
            name=spec.name, category=spec.category, status="error",
            error=str(e),
            repairable=True,
        )


def _setup_hf_file_model(spec: ModelSpec, repair: bool = False) -> ModelStatus:
    repo_id = os.environ.get(spec.env_var or "", spec.default_id)
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
        logger.info("Downloading %s from %s...", spec.name, repo_id)
        path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            force_download=repair,
        )
        logger.info("%s ready: %s", spec.name, path)
        return ModelStatus(
            name=spec.name, category=spec.category, status="ok",
            path=path,
        )
    except ImportError:
        return ModelStatus(
            name=spec.name, category=spec.category, status="error",
            error="huggingface_hub not installed. Run: uv sync --extra real",
        )
    except Exception as e:
        return ModelStatus(
            name=spec.name, category=spec.category, status="error",
            error=str(e),
            repairable=True,
        )


def _setup_hf_repo_model(spec: ModelSpec, repair: bool = False) -> ModelStatus:
    repo_id = os.environ.get(spec.env_var or "", spec.default_id)
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
        logger.info("Downloading %s snapshot from %s...", spec.name, repo_id)
        path = snapshot_download(
            repo_id=repo_id,
            force_download=repair,
        )
        logger.info("%s ready: %s", spec.name, path)
        return ModelStatus(
            name=spec.name, category=spec.category, status="ok",
            path=path,
        )
    except ImportError:
        return ModelStatus(
            name=spec.name, category=spec.category, status="error",
            error="huggingface_hub not installed. Run: uv sync --extra real",
        )
    except Exception as e:
        return ModelStatus(
            name=spec.name, category=spec.category, status="error",
            error=str(e),
            repairable=True,
        )


def setup_model(spec: ModelSpec, repair: bool = False) -> ModelStatus:
    if spec.model_type == "whisper":
        return _setup_whisper_model(spec)
    elif spec.model_type == "hf_file":
        return _setup_hf_file_model(spec, repair=repair)
    elif spec.model_type == "hf_repo":
        return _setup_hf_repo_model(spec, repair=repair)
    return ModelStatus(
        name=spec.name, category=spec.category, status="error",
        error=f"Unknown model type: {spec.model_type}",
    )


def setup_models(
    category: str | None = None,
    repair: bool = False,
) -> SetupReport:
    specs = _model_specs()
    if category:
        specs = [s for s in specs if s.category == category]

    report = SetupReport()
    for spec in specs:
        status = setup_model(spec, repair=repair)
        report.models.append(status)
        if status.status == "error":
            report.errors.append(f"{status.name}: {status.error}")

    return report


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def repair_models(category: str | None = None) -> SetupReport:
    """Re-download models that are missing or broken."""
    return setup_models(category=category, repair=True)


# ---------------------------------------------------------------------------
# CLI display helpers
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "ok": "OK",
    "missing": "MISSING",
    "error": "ERROR",
    "fixable": "FIXABLE",
}


def print_model_status(status: ModelStatus) -> None:
    icon = _STATUS_ICONS.get(status.status, status.status.upper())
    line = f"  [{icon}] {status.name}"
    if status.path:
        line += f" → {status.path}"
    if status.error:
        line += f" ({status.error})"
    print(line)


def print_setup_report(report: SetupReport, title: str = "Model Status") -> None:
    print(title)
    print("=" * 40)
    for status in report.models:
        print_model_status(status)
    print()
    if report.errors:
        print("Errors:")
        for err in report.errors:
            print(f"  - {err}")
        print()
