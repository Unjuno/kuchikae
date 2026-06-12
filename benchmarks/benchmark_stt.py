"""Benchmark the STT backend."""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_utils import (
    audio_duration_sec,
    ensure_dir,
    git_sha,
    machine_info,
    memory_mb,
    median_value,
    percentile,
    repo_root,
    write_wav,
)
from kuchikae.backends.stt import FasterWhisperSTTBackend
from kuchikae.backends.stt_ct2 import AnimeWhisperCT2FP16STTBackend, AnimeWhisperCT2STTBackend
from kuchikae.backends.stt_nemo import ReazonSpeechNemoASRBackend
from kuchikae.backends.stt_transformers import TransformersJapaneseASRBackend
from kuchikae.backends.stt_transformers_whisper import TransformersWhisperJapaneseASRBackend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--audio-seconds", type=float, default=5.0)
    parser.add_argument("--audio-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/stt.json"))
    parser.add_argument("--model-size", default=None)
    parser.add_argument(
        "--backend",
        choices=[
            "faster_whisper",
            "anime_whisper_ct2",
            "anime_whisper_ct2_fp16",
            "transformers_japanese",
            "transformers_whisper",
            "reazonspeech_nemo",
        ],
        default="faster_whisper",
    )
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()

    ensure_dir(args.output.parent)
    if args.audio_path is not None:
        tmp_audio = args.audio_path
    else:
        tmp_audio = ensure_dir(repo_root() / ".bench_audio") / "stt.wav"
        write_wav(tmp_audio, duration_sec=args.audio_seconds)

    if args.backend == "transformers_japanese":
        backend = TransformersJapaneseASRBackend(model_id=args.model_id)
    elif args.backend == "transformers_whisper":
        backend = TransformersWhisperJapaneseASRBackend(model_id=args.model_id)
    elif args.backend == "reazonspeech_nemo":
        backend = ReazonSpeechNemoASRBackend(model_id=args.model_id)
    elif args.backend == "anime_whisper_ct2":
        backend = AnimeWhisperCT2STTBackend()
    elif args.backend == "anime_whisper_ct2_fp16":
        backend = AnimeWhisperCT2FP16STTBackend()
    elif args.model_size:
        backend = FasterWhisperSTTBackend(model_size=args.model_size)
    else:
        backend = FasterWhisperSTTBackend()

    load_t0 = time.perf_counter()
    try:
        backend._load_model()  # noqa: SLF001
        load_sec = time.perf_counter() - load_t0
    except Exception as e:
        result = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_sha": git_sha(),
            "machine": machine_info(),
            "backend": {
                "stt": type(backend).__name__,
                "model_id": getattr(backend, "_model_id", None) or getattr(backend, "_model_size", None),
                "device": getattr(backend, "_device", None),
                "compute_type": getattr(backend, "_compute_type", None) or getattr(backend, "_torch_dtype", None),
                "config": {},
            },
            "runs": [{"name": "cold", "load_sec": 0.0, "inference_sec": 0.0, "total_sec": 0.0, "rtf": 0.0, "memory_mb": memory_mb(), "text": "", "text_len": 0, "output_path": "", "output_exists": False, "output_duration_sec": 0.0, "error": f"{type(e).__name__}: {e}"}],
            "summary": {"median_warm_total_sec": 0.0, "median_warm_rtf": 0.0, "p95_warm_total_sec": 0.0},
        }
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(args.output)
        return 0

    cold_t0 = time.perf_counter()
    try:
        text = backend.transcribe(str(tmp_audio))
        cold_sec = time.perf_counter() - cold_t0
        runs = [{
            "name": "cold",
            "load_sec": load_sec,
            "inference_sec": cold_sec,
            "total_sec": load_sec + cold_sec,
            "rtf": cold_sec / max(audio_duration_sec(str(tmp_audio)), 1e-6),
            "memory_mb": memory_mb(),
            "text": text,
            "text_len": len(text),
            "output_path": "",
            "output_exists": False,
            "output_duration_sec": 0.0,
            "error": None,
        }]
    except Exception as e:
        runs = [{
            "name": "cold",
            "load_sec": load_sec,
            "inference_sec": time.perf_counter() - cold_t0,
            "total_sec": load_sec + (time.perf_counter() - cold_t0),
            "rtf": None,
            "memory_mb": memory_mb(),
            "text": "",
            "text_len": 0,
            "output_path": "",
            "output_exists": False,
            "output_duration_sec": 0.0,
            "error": f"{type(e).__name__}: {e}",
        }]

    warm_totals = []
    warm_rtfs = []
    for idx in range(args.warmup):
        _ = backend.transcribe(str(tmp_audio))
    for idx in range(args.runs):
        t0 = time.perf_counter()
        text = backend.transcribe(str(tmp_audio))
        sec = time.perf_counter() - t0
        warm_totals.append(sec)
        warm_rtfs.append(sec / max(audio_duration_sec(str(tmp_audio)), 1e-6))
        runs.append({
            "name": f"warm{idx+1}",
            "load_sec": 0.0,
            "inference_sec": sec,
            "total_sec": sec,
            "rtf": sec / max(audio_duration_sec(str(tmp_audio)), 1e-6),
            "memory_mb": memory_mb(),
            "text": text,
            "text_len": len(text),
            "output_path": "",
            "output_exists": False,
            "output_duration_sec": 0.0,
            "error": None,
        })

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "machine": machine_info(),
        "backend": {
            "stt": type(backend).__name__,
            "model_id": getattr(backend, "_model_id", None) or getattr(backend, "_model_size", None),
            "device": getattr(backend, "_device", None),
            "compute_type": getattr(backend, "_compute_type", None) or getattr(backend, "_torch_dtype", None),
            "config": {},
        },
        "runs": runs,
        "summary": {
            "median_warm_total_sec": median_value(warm_totals),
            "median_warm_rtf": median_value(warm_rtfs),
            "p95_warm_total_sec": percentile(warm_totals, 95),
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
