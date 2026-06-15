#!/usr/bin/env python3
"""Performance benchmark for Kuchikae pipeline.

Measures end-to-end latency, per-component latency, and realtime factor
for both batch and streaming pipelines.

Usage:
    uv run python scripts/benchmark_performance.py
    uv run python scripts/benchmark_performance.py --iterations 5
    uv run python scripts/benchmark_performance.py --component stt
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass

import numpy as np
import soundfile as sf


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    component: str
    iterations: int
    total_time_sec: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    realtime_factor: float | None = None
    audio_duration_sec: float | None = None


def create_test_audio(path: str, duration_sec: float = 5.0, sr: int = 16000) -> float:
    """Create a test WAV file with speech-like content."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Simulate speech with fundamental + harmonics
    signal = 0.1 * np.sin(2 * np.pi * 150 * t)  # Fundamental
    signal += 0.05 * np.sin(2 * np.pi * 300 * t)  # 1st harmonic
    signal += 0.03 * np.sin(2 * np.pi * 450 * t)  # 2nd harmonic
    # Add some amplitude modulation to simulate speech rhythm
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    signal *= envelope
    sf.write(path, signal.astype(np.float32), sr)
    return duration_sec


def benchmark_stt(iterations: int = 3) -> BenchmarkResult | None:
    """Benchmark STT component."""
    try:
        from kuchikae.backends.stt import FasterWhisperSTTBackend
    except ImportError as e:
        print(f"  [SKIP] faster-whisper not installed: {e}")
        return None

    backend = FasterWhisperSTTBackend(model_size="tiny")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name
    try:
        create_test_audio(audio_path, duration_sec=3.0)

        # Warmup
        backend.transcribe(audio_path)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            backend.transcribe(audio_path)
            latencies.append(time.perf_counter() - t0)

        os.unlink(audio_path)
        audio_dur = 3.0
    except Exception:
        os.unlink(audio_path)
        raise
    return BenchmarkResult(
        component="STT (FasterWhisper)",
        iterations=iterations,
        total_time_sec=sum(latencies),
        avg_latency_ms=np.mean(latencies) * 1000,
        min_latency_ms=min(latencies) * 1000,
        max_latency_ms=max(latencies) * 1000,
        p95_latency_ms=np.percentile(latencies, 95) * 1000,
        realtime_factor=sum(latencies) / audio_dur,
        audio_duration_sec=audio_dur,
    )


def benchmark_text_transform(iterations: int = 3) -> BenchmarkResult | None:
    """Benchmark text transform component."""
    try:
        from kuchikae.domain.text_transform import OllamaTextTransformBackend
        from kuchikae.domain.types import TextTransformPrompt
    except ImportError as e:
        print(f"  [SKIP] text transform backend not available: {e}")
        return None

    backend = OllamaTextTransformBackend()
    prompt = TextTransformPrompt(instruction="丁寧な日本語にしてください")
    test_text = "明日3時に資料を2つ送って"

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        backend.transform(test_text, prompt)
        latencies.append(time.perf_counter() - t0)

    return BenchmarkResult(
        component="TextTransform (Ollama)",
        iterations=iterations,
        total_time_sec=sum(latencies),
        avg_latency_ms=np.mean(latencies) * 1000,
        min_latency_ms=min(latencies) * 1000,
        max_latency_ms=max(latencies) * 1000,
        p95_latency_ms=np.percentile(latencies, 95) * 1000,
    )


def benchmark_tts(iterations: int = 3) -> BenchmarkResult | None:
    """Benchmark TTS component."""
    try:
        from kuchikae.domain.voice_output import DummyVoiceOutputBackend
        from kuchikae.domain.types import VoiceContext
    except ImportError as e:
        print(f"  [SKIP] TTS backend not available: {e}")
        return None

    backend = DummyVoiceOutputBackend()
    vc = VoiceContext(reference_audio_path="", ready=False)
    test_text = "テストです"

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = backend.synthesize(test_text, vc)
        latencies.append(time.perf_counter() - t0)
        if os.path.exists(result):
            os.unlink(result)

    return BenchmarkResult(
        component="TTS (Dummy)",
        iterations=iterations,
        total_time_sec=sum(latencies),
        avg_latency_ms=np.mean(latencies) * 1000,
        min_latency_ms=min(latencies) * 1000,
        max_latency_ms=max(latencies) * 1000,
        p95_latency_ms=np.percentile(latencies, 95) * 1000,
    )


def benchmark_pipeline(iterations: int = 3) -> BenchmarkResult | None:
    """Benchmark full pipeline."""
    try:
        from kuchikae.pipeline.pipeline import KuchikaePipeline
        from kuchikae.domain.types import TextTransformPrompt
    except ImportError as e:
        print(f"  [SKIP] pipeline not available: {e}")
        return None

    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="丁寧な日本語にしてください")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name
    try:
        create_test_audio(audio_path, duration_sec=2.0)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            result = pipeline.process(audio_path, prompt)
            latencies.append(time.perf_counter() - t0)
            if os.path.exists(result.output_audio_path):
                os.unlink(result.output_audio_path)

        os.unlink(audio_path)
        audio_dur = 2.0
    except Exception:
        os.unlink(audio_path)
        raise
    return BenchmarkResult(
        component="Pipeline (Full)",
        iterations=iterations,
        total_time_sec=sum(latencies),
        avg_latency_ms=np.mean(latencies) * 1000,
        min_latency_ms=min(latencies) * 1000,
        max_latency_ms=max(latencies) * 1000,
        p95_latency_ms=np.percentile(latencies, 95) * 1000,
        realtime_factor=sum(latencies) / audio_dur,
        audio_duration_sec=audio_dur,
    )


def print_result(result: BenchmarkResult) -> None:
    """Pretty-print a benchmark result."""
    print(f"\n{'='*60}")
    print(f"  {result.component}")
    print(f"{'='*60}")
    print(f"  Iterations:      {result.iterations}")
    print(f"  Total time:      {result.total_time_sec:.2f}s")
    print(f"  Avg latency:     {result.avg_latency_ms:.1f}ms")
    print(f"  Min latency:     {result.min_latency_ms:.1f}ms")
    print(f"  Max latency:     {result.max_latency_ms:.1f}ms")
    print(f"  P95 latency:     {result.p95_latency_ms:.1f}ms")
    if result.realtime_factor is not None:
        print(f"  Realtime factor: {result.realtime_factor:.2f}x")
    if result.audio_duration_sec is not None:
        print(f"  Audio duration:  {result.audio_duration_sec:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Kuchikae performance benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=3,
                        help="Number of iterations (default: 3)")
    parser.add_argument("--component", "-c", type=str, default="all",
                        choices=["all", "stt", "text", "tts", "pipeline"],
                        help="Component to benchmark (default: all)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()

    print("=" * 60)
    print("  Kuchikae Performance Benchmark")
    print("=" * 60)
    print(f"  Iterations: {args.iterations}")
    print(f"  Component:  {args.component}")

    results: list[BenchmarkResult] = []

    benchmarks = {
        "stt": ("STT (FasterWhisper)", benchmark_stt),
        "text": ("TextTransform (Ollama)", benchmark_text_transform),
        "tts": ("TTS (Dummy)", benchmark_tts),
        "pipeline": ("Pipeline (Full)", benchmark_pipeline),
    }

    if args.component == "all":
        for name, (label, func) in benchmarks.items():
            print(f"\nBenchmarking {label}...")
            result = func(args.iterations)
            if result is not None:
                results.append(result)
                print_result(result)
    else:
        label, func = benchmarks[args.component]
        print(f"\nBenchmarking {label}...")
        result = func(args.iterations)
        if result is not None:
            results.append(result)
            print_result(result)

    # Summary
    if results:
        print(f"\n{'='*60}")
        print("  Summary")
        print(f"{'='*60}")
        for r in results:
            rf = f" ({r.realtime_factor:.2f}x)" if r.realtime_factor else ""
            print(f"  {r.component}: {r.avg_latency_ms:.1f}ms{rf}")

    # Save results
    if args.output and results:
        output_data = [asdict(r) for r in results]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
