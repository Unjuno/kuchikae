"""Shared helpers for benchmark scripts."""

from __future__ import annotations

import json
import logging
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG = logging.getLogger(__name__)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True).strip()
    except Exception:
        return "unknown"


def machine_info() -> dict[str, Any]:
    info = platform.uname()
    torch_info: dict[str, Any] = {"installed": False}
    try:
        import torch

        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception:
        LOG.warning("torch not available", exc_info=True)
        torch_info = {"installed": False}

    return {
        "platform": sys.platform,
        "machine": info.machine,
        "processor": info.processor,
        "python": sys.version.split()[0],
        "torch": torch_info,
        "mps_available": torch_info.get("mps_available", False),
        "cuda_available": torch_info.get("cuda_available", False),
        "macos_version": platform.mac_ver()[0],
        "ffmpeg": shutil.which("ffmpeg"),
        "uv_env": bool(os.environ.get("VIRTUAL_ENV") or os.environ.get("UV_PROJECT_ENVIRONMENT")),
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_wav(path: Path, duration_sec: float = 5.0, sample_rate: int = 16000, freq: float = 220.0) -> Path:
    samples = np.zeros(int(duration_sec * sample_rate), dtype=np.float32)
    if duration_sec > 0:
        t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
        samples = 0.04 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), samples, sample_rate)
    return path


def audio_duration_sec(path: str) -> float:
    return float(sf.info(path).duration)


def memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, p))


def median_value(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(median(values))


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value
