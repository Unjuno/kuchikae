"""Benchmark the full Kuchikae pipeline."""

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
    memory_mb,
    median_value,
    percentile,
    repo_root,
    machine_info,
    write_wav,
)
from kuchikae.domain.types import TextTransformPrompt, VoiceOutputPrompt
from kuchikae.pipeline import create_pipeline


def run_case(pipeline, audio_path: str, text_prompt: TextTransformPrompt, voice_prompt: VoiceOutputPrompt | None) -> dict:
    t0 = time.perf_counter()
    try:
        result = pipeline.process(audio_path, text_prompt, voice_prompt)
        total = time.perf_counter() - t0
        output_exists = Path(result.output_audio_path).exists()
        duration = audio_duration_sec(result.output_audio_path) if output_exists else 0.0
        return {
            "load_sec": 0.0,
            "inference_sec": total,
            "total_sec": total,
            "rtf": total / max(audio_duration_sec(audio_path), 1e-6),
            "memory_mb": memory_mb(),
            "text": result.source_text,
            "text_len": len(result.source_text),
            "output_path": result.output_audio_path,
            "output_exists": output_exists,
            "output_duration_sec": duration,
            "error": None,
            "stt_latency": result.stt_latency,
            "text_transform_latency": result.text_transform_latency,
            "voice_output_latency": result.voice_output_latency,
        }
    except Exception as e:
        return {
            "load_sec": 0.0,
            "inference_sec": time.perf_counter() - t0,
            "total_sec": time.perf_counter() - t0,
            "rtf": None,
            "memory_mb": memory_mb(),
            "text": "",
            "text_len": 0,
            "output_path": "",
            "output_exists": False,
            "output_duration_sec": 0.0,
            "error": f"{type(e).__name__}: {e}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--audio-seconds", type=float, default=5.0)
    parser.add_argument("--audio-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/pipeline.json"))
    parser.add_argument("--clear-cache-between-runs", action="store_true", default=True)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--allow-dummy-backends", action="store_true", default=True)
    parser.add_argument("--streaming-stt", action="store_true")
    parser.add_argument("--segmented-stt", action="store_true")
    parser.add_argument("--stt-backend", default="faster_whisper")
    parser.add_argument("--text-backend", default="prompted_rule")
    parser.add_argument("--voice-backend", default="irodori")
    args = parser.parse_args()

    ensure_dir(args.output.parent)
    if args.audio_path is not None:
        tmp_audio = args.audio_path
    else:
        tmp_audio = ensure_dir(repo_root() / ".bench_audio") / "sample.wav"
        write_wav(tmp_audio, duration_sec=args.audio_seconds)

    pipeline = create_pipeline(
        {
            "allow_dummy_backends": args.allow_dummy_backends,
            "streaming_stt": args.streaming_stt,
            "segmented_stt": args.segmented_stt,
            "stt_backend": args.stt_backend,
            "text_transform_backend": args.text_backend,
            "voice_output_backend": args.voice_backend,
        }
    )
    pipeline.warmup()

    text_prompt = TextTransformPrompt(instruction="内容、数字、日時、固有名詞、否定条件は保ちつつ、言い回しを自然な日本語に変換してください。")
    voice_prompt = VoiceOutputPrompt(instruction="落ち着いた自然な読み上げ")

    runs = []
    for _ in range(args.warmup):
        if args.clear_cache_between_runs and not args.keep_cache:
            pipeline.processing_cache.clear()
        _ = run_case(pipeline, str(tmp_audio), text_prompt, voice_prompt)
    for idx in range(args.runs):
        if args.clear_cache_between_runs and not args.keep_cache:
            pipeline.processing_cache.clear()
        runs.append({"name": f"run{idx+1}", **run_case(pipeline, str(tmp_audio), text_prompt, voice_prompt)})

    warm_totals = [r["total_sec"] for r in runs if isinstance(r["total_sec"], (int, float))]
    warm_rtfs = [r["rtf"] for r in runs if isinstance(r["rtf"], (int, float))]
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "machine": machine_info(),
        "backend": {
            "stt": type(pipeline.stt_backend).__name__,
            "tts": type(pipeline.voice_output_backend).__name__,
            "model_id": getattr(pipeline.voice_output_backend, "_hf_checkpoint", None),
            "device": getattr(pipeline.stt_backend, "_device", None),
            "compute_type": getattr(pipeline.stt_backend, "_compute_type", None),
            "config": {
                "streaming_stt": args.streaming_stt,
                "segmented_stt": args.segmented_stt,
                "text_backend": args.text_backend,
                "voice_backend": args.voice_backend,
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
