"""Benchmark the TTS backend."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

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
)
from kuchikae.backends.voice_output import DummyVoiceOutputBackend, IrodoriTTSVoiceOutputBackend
from kuchikae.domain.types import VoiceContext


def _write_ref(path: Path, sr: int = 24000, duration_sec: float = 2.0) -> Path:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    signal = 0.08 * np.sin(2 * np.pi * 220 * t)
    sf.write(str(path), signal.astype(np.float32), sr)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/tts.json"))
    parser.add_argument("--backend", choices=["irodori", "dummy"], default="irodori")
    parser.add_argument("--text", default="明日までに資料を送ってください。")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--cfg-scale-text", type=float, default=None)
    parser.add_argument("--cfg-scale-speaker", type=float, default=None)
    parser.add_argument("--cfg-guidance-mode", default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--cfg-min-t", type=float, default=None)
    parser.add_argument("--cfg-max-t", type=float, default=None)
    parser.add_argument("--speaker-kv-scale", type=float, default=None)
    parser.add_argument("--speaker-kv-min-t", type=float, default=None)
    parser.add_argument("--speaker-kv-max-layers", type=int, default=None)
    parser.add_argument("--speaker-uncond-mode", default=None)
    parser.add_argument("--t-schedule-mode", default=None)
    parser.add_argument("--sway-coeff", type=float, default=None)
    parser.add_argument("--ref-normalize-db", type=float, default=None)
    parser.add_argument("--ref-ensure-max", action="store_true", default=False)
    parser.add_argument("--no-ref-ensure-max", action="store_true", default=False)
    parser.add_argument("--decode-mode", default=None)
    parser.add_argument("--duration-scale", type=float, default=None)
    parser.add_argument("--min-seconds", type=float, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--max-ref-seconds", type=float, default=None)
    parser.add_argument("--max-text-len", type=int, default=None)
    parser.add_argument("--max-caption-len", type=int, default=None)
    parser.add_argument("--truncation-factor", type=float, default=None)
    parser.add_argument("--rescale-k", type=float, default=None)
    parser.add_argument("--rescale-sigma", type=float, default=None)
    parser.add_argument("--tail-window-size", type=int, default=None)
    parser.add_argument("--tail-std-threshold", type=float, default=None)
    parser.add_argument("--tail-mean-threshold", type=float, default=None)
    parser.add_argument("--lora-adapter", default=None)
    args = parser.parse_args()

    ensure_dir(args.output.parent)
    ref = ensure_dir(Path(tempfile.gettempdir()) / "kuchikae-bench") / "ref.wav"
    _write_ref(ref)
    vc = VoiceContext(reference_audio_path=str(ref), ready=True)

    backend = (
        IrodoriTTSVoiceOutputBackend(
            num_steps=args.num_steps or 10,
            cfg_scale_text=args.cfg_scale_text or 2.0,
            cfg_scale_speaker=args.cfg_scale_speaker or 3.0,
            cfg_guidance_mode=args.cfg_guidance_mode or "independent",
            cfg_scale=args.cfg_scale,
            cfg_min_t=args.cfg_min_t or 0.5,
            cfg_max_t=args.cfg_max_t or 1.0,
            speaker_kv_scale=args.speaker_kv_scale,
            speaker_kv_min_t=args.speaker_kv_min_t,
            speaker_kv_max_layers=args.speaker_kv_max_layers,
            speaker_uncond_mode=args.speaker_uncond_mode or "mask",
            t_schedule_mode=args.t_schedule_mode or "linear",
            sway_coeff=args.sway_coeff if args.sway_coeff is not None else -1.0,
            ref_normalize_db=args.ref_normalize_db if args.ref_normalize_db is not None else -16.0,
            ref_ensure_max=True if args.ref_ensure_max else (False if args.no_ref_ensure_max else True),
            decode_mode=args.decode_mode or "sequential",
            duration_scale=args.duration_scale or 1.0,
            min_seconds=args.min_seconds or 0.5,
            max_seconds=args.max_seconds or 30.0,
            max_ref_seconds=args.max_ref_seconds or 30.0,
            max_text_len=args.max_text_len,
            max_caption_len=args.max_caption_len,
            truncation_factor=args.truncation_factor,
            rescale_k=args.rescale_k,
            rescale_sigma=args.rescale_sigma,
            tail_window_size=args.tail_window_size or 20,
            tail_std_threshold=args.tail_std_threshold or 0.05,
            tail_mean_threshold=args.tail_mean_threshold or 0.1,
            lora_adapter=args.lora_adapter,
        )
        if args.backend == "irodori"
        else DummyVoiceOutputBackend()
    )

    load_t0 = time.perf_counter()
    try:
        if hasattr(backend, "_ensure_runtime"):
            backend._ensure_runtime()  # noqa: SLF001
        load_sec = time.perf_counter() - load_t0
    except Exception as e:
        load_sec = time.perf_counter() - load_t0
        result = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_sha": git_sha(),
            "machine": machine_info(),
            "backend": {"stt": "", "tts": type(backend).__name__, "model_id": getattr(backend, "_hf_checkpoint", None), "device": None, "compute_type": None, "config": {}},
            "runs": [{"name": "cold", "load_sec": load_sec, "inference_sec": 0.0, "total_sec": load_sec, "rtf": None, "memory_mb": memory_mb(), "text": "", "text_len": 0, "output_path": "", "output_exists": False, "output_duration_sec": 0.0, "error": f"{type(e).__name__}: {e}"}],
            "summary": {"median_warm_total_sec": 0.0, "median_warm_rtf": 0.0, "p95_warm_total_sec": 0.0},
        }
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(args.output)
        return 0

    runs = []
    output_durations = []
    warm_totals = []
    warm_rtfs = []

    # cold
    cold_t0 = time.perf_counter()
    try:
        output = backend.synthesize(args.text, vc)
        cold_sec = time.perf_counter() - cold_t0
        duration = audio_duration_sec(output) if Path(output).exists() else 0.0
        output_durations.append(duration)
        runs.append({
            "name": "cold",
            "load_sec": load_sec,
            "inference_sec": cold_sec,
            "total_sec": load_sec + cold_sec,
            "rtf": cold_sec / max(duration, 1e-6) if duration > 0 else None,
            "memory_mb": memory_mb(),
            "text": args.text,
            "text_len": len(args.text),
            "output_path": output,
            "output_exists": Path(output).exists(),
            "output_duration_sec": duration,
            "error": None,
        })
    except Exception as e:
        runs.append({
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
        })

    for _ in range(args.warmup):
        _ = backend.synthesize(args.text, vc)
    for idx in range(args.runs):
        t0 = time.perf_counter()
        output = backend.synthesize(args.text, vc)
        sec = time.perf_counter() - t0
        duration = audio_duration_sec(output) if Path(output).exists() else 0.0
        warm_totals.append(sec)
        warm_rtfs.append(sec / max(duration, 1e-6) if duration > 0 else 0.0)
        output_durations.append(duration)
        runs.append({
            "name": f"warm{idx+1}",
            "load_sec": 0.0,
            "inference_sec": sec,
            "total_sec": sec,
            "rtf": sec / max(duration, 1e-6) if duration > 0 else None,
            "memory_mb": memory_mb(),
            "text": args.text,
            "text_len": len(args.text),
            "output_path": output,
            "output_exists": Path(output).exists(),
            "output_duration_sec": duration,
            "error": None,
        })

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "machine": machine_info(),
        "backend": {
            "stt": "",
            "tts": type(backend).__name__,
            "model_id": getattr(backend, "_hf_checkpoint", None),
            "device": None,
            "compute_type": None,
            "config": {
                "backend": args.backend,
                "num_steps": args.num_steps or 10,
                "cfg_scale_text": args.cfg_scale_text or 2.0,
                "cfg_scale_speaker": args.cfg_scale_speaker or 3.0,
                "cfg_guidance_mode": args.cfg_guidance_mode or "independent",
                "cfg_scale": args.cfg_scale,
                "cfg_min_t": args.cfg_min_t or 0.5,
                "cfg_max_t": args.cfg_max_t or 1.0,
                "speaker_kv_scale": args.speaker_kv_scale,
                "speaker_kv_min_t": args.speaker_kv_min_t,
                "speaker_kv_max_layers": args.speaker_kv_max_layers,
                "speaker_uncond_mode": args.speaker_uncond_mode or "mask",
                "t_schedule_mode": args.t_schedule_mode or "linear",
                "sway_coeff": args.sway_coeff if args.sway_coeff is not None else -1.0,
                "ref_normalize_db": args.ref_normalize_db if args.ref_normalize_db is not None else -16.0,
                "ref_ensure_max": True if args.ref_ensure_max else (False if args.no_ref_ensure_max else True),
                "decode_mode": args.decode_mode or "sequential",
                "duration_scale": args.duration_scale or 1.0,
                "min_seconds": args.min_seconds or 0.5,
                "max_seconds": args.max_seconds or 30.0,
                "max_ref_seconds": args.max_ref_seconds or 30.0,
                "max_text_len": args.max_text_len,
                "max_caption_len": args.max_caption_len,
                "truncation_factor": args.truncation_factor,
                "rescale_k": args.rescale_k,
                "rescale_sigma": args.rescale_sigma,
                "tail_window_size": args.tail_window_size or 20,
                "tail_std_threshold": args.tail_std_threshold or 0.05,
                "tail_mean_threshold": args.tail_mean_threshold or 0.1,
                "lora_adapter": args.lora_adapter,
            },
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
