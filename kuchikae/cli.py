"""Kuchikae CLI entry point."""

from __future__ import annotations

import os
import sys
import platform


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        print_help()
        return

    if args[0] == "serve":
        cmd_serve(args[1:])
        return

    if args[0] == "doctor":
        cmd_doctor(args[1:])
        return

    if args[0] == "setup-models":
        cmd_setup_models(args[1:])
        return

    # Backward compatibility: bare --streaming is a compatibility alias
    # for serve --real --streaming (will fail if real deps are missing)
    if "--streaming" in args:
        print("Warning: 'kuchikae --streaming' is deprecated. Use 'kuchikae serve --real --streaming' instead.", file=sys.stderr)
        from kuchikae.web import serve
        serve(real=True, streaming=True)
        return

    print(f"Unknown command: {args[0]}")
    print("Run 'kuchikae --help' for usage.")
    sys.exit(1)


def print_help() -> None:
    port = os.environ.get("GRADIO_SERVER_PORT", "7860")
    print("""Kuchikae - Prompt-conditioned speech transformation

Usage:
  kuchikae serve [options]          Start the web server
  kuchikae doctor [--fix]           Check backend availability
  kuchikae setup-models [options]   Download required model weights
  kuchikae --help                   Show this help

Options:
  --dummy         Use dummy backends for smoke testing
  --real          Use real backends (requires models)
  --streaming     Enable streaming STT with --real
  --port PORT     Server port (default: 7860)
  --text-model    Text transform model (default: gemma3:1b-it-qat)

Setup:
  setup-models               Download all required models
  setup-models --all         Download all models (including optional)
  setup-models --stt         Download STT model only
  setup-models --tts         Download TTS model only
  setup-models --emotion     Download audio emotion model only
  setup-models --all --repair  Re-download (force) all models

Doctor:
  doctor              Check backend and model status
  doctor --fix        Attempt to repair missing/broken models
  doctor --strict     Exit with non-zero status on errors

Examples:
  kuchikae serve --dummy              # Smoke test with dummy backends
  kuchikae setup-models --all         # Download all models
  kuchikae doctor --fix               # Check and repair models
  kuchikae serve --real               # Use real STT/TTS backends
  kuchikae serve --port 8080          # Use custom port
  kuchikae serve --text-model qwen2.5:7b-instruct  # Use specific model""")

    print(f"\nWeb UI will be available at http://127.0.0.1:{port}")


def cmd_serve(args: list[str]) -> None:
    dummy = "--dummy" in args
    real = "--real" in args
    streaming = "--streaming" in args

    port = None
    text_model = None
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                print(f"Invalid port: {args[i + 1]}")
                sys.exit(1)
        if arg == "--text-model" and i + 1 < len(args):
            text_model = args[i + 1]

    if real and dummy:
        print("Cannot use both --real and --dummy.")
        sys.exit(1)

    # Set text model environment variable if specified
    if text_model:
        os.environ["KUCHIKAE_TEXT_MODEL"] = text_model

    from kuchikae.web import serve
    serve(dummy=dummy, real=real, streaming=streaming, port=port)


def cmd_setup_models(args: list[str]) -> None:
    if "--help" in args or "-h" in args:
        print_setup_models_help()
        return

    all_models = "--all" in args
    repair = "--repair" in args
    category = None

    if "--stt" in args:
        category = "stt"
    elif "--tts" in args:
        category = "tts"
    elif "--emotion" in args:
        category = "emotion"
    elif not all_models:
        # Default: required models only (stt + tts)
        category = None

    from kuchikae.models import setup_models, print_setup_report

    if all_models:
        # Include optional models too
        report = setup_models(category=None, repair=repair)
    else:
        report = setup_models(category=category, repair=repair)

    print_setup_report(report, title="Setup Results")

    if report.errors:
        sys.exit(1)


def print_setup_models_help() -> None:
    print("""Usage: kuchikae setup-models [options]

Download required model weights for Kuchikae's real backends.

Options:
  --all       Download all models (including optional audio emotion)
  --stt       Download STT model only (FasterWhisper)
  --tts       Download TTS models only (Irodori-TTS + codec)
  --emotion   Download audio emotion model only (optional)
  --repair    Force re-download even if cached
  --help      Show this help

Examples:
  kuchikae setup-models           # Download required models (stt + tts)
  kuchikae setup-models --all     # Download all models
  kuchikae setup-models --stt     # Download STT model only
  kuchikae setup-models --repair  # Re-download all models""")


def cmd_doctor(args: list[str]) -> None:
    strict = "--strict" in args
    fix = "--fix" in args

    print("Kuchikae Doctor")
    print("=" * 40)

    # Python version
    print(f"\nPython: {platform.python_version()}")

    # Package version
    try:
        from kuchikae import __version__
        print(f"Package: {__version__}")
    except ImportError:
        print("Package: not installed")

    # Environment variables
    print("\nEnvironment Variables:")
    env_vars = [
        "KUCHIKAE_STT_BACKEND",
        "KUCHIKAE_TEXT_BACKEND",
        "KUCHIKAE_VOICE_BACKEND",
        "KUCHIKAE_ALLOW_DUMMY_BACKENDS",
        "KUCHIKAE_STREAMING_STT",
        "KUCHIKAE_STT_PRESET",
        "KUCHIKAE_TEXT_MODEL",
        "OPENAI_API_KEY",
        "KUCHIKAE_OPENVOICE_READY",
    ]
    for var in env_vars:
        value = os.environ.get(var, "(not set)")
        if value and len(value) > 30:
            value = value[:30] + "..."
        print(f"  {var}: {value}")

    # Core dependencies
    print("\nCore dependencies:")
    core_deps = [
        ("gradio", "gradio"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("httpx", "httpx"),
    ]
    core_ok = True
    for module_name, package_name in core_deps:
        try:
            __import__(module_name)
            print(f"  {package_name}: OK")
        except ImportError:
            print(f"  {package_name}: NOT INSTALLED")
            core_ok = False

    # Optional real backends
    print("\nOptional real backends:")
    real_deps = [
        ("faster_whisper", "faster-whisper"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("irodori_tts", "irodori-tts"),
    ]
    for module_name, package_name in real_deps:
        try:
            __import__(module_name)
            print(f"  {package_name}: OK")
        except ImportError:
            print(f"  {package_name}: not installed")

    # Check Ollama
    print("\nOllama:")
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print(f"  Status: Running ({len(models)} models)")
            for m in models[:5]:
                print(f"    - {m.get('name', 'unknown')}")
        else:
            print(f"  Status: Error (HTTP {resp.status_code})")
    except Exception as e:
        print(f"  Status: Not reachable ({type(e).__name__})")

    # Text model warning
    text_model = os.environ.get("KUCHIKAE_TEXT_MODEL", "")
    if text_model:
        if "coder" in text_model.lower():
            print(f"\n  Warning: '{text_model}' is a coder model, not recommended for text transform.")
            print("  Use an instruct model instead (e.g. qwen2.5:7b-instruct).")
        if "instruct" not in text_model.lower() and "chat" not in text_model.lower():
            print(f"\n  Warning: '{text_model}' may not be an instruct model.")
            print("  Non-instruct models may emit CoT/extra text and respond slowly.")

    # Model status
    print("\nModels:")
    try:
        from kuchikae.models import check_models, print_model_status
        model_statuses = check_models()
        any_missing = False
        any_error = False
        for status in model_statuses:
            print_model_status(status)
            if status.status == "missing":
                any_missing = True
            if status.status == "error":
                any_error = True

        # Fix mode: attempt repair
        if fix and (any_missing or any_error):
            print("\nAttempting repair...")
            from kuchikae.models import repair_models
            repair_report = repair_models()
            for status in repair_report.models:
                print_model_status(status)
            if repair_report.errors:
                print("\nRepair completed with errors:")
                for err in repair_report.errors:
                    print(f"  - {err}")
    except Exception as e:
        print(f"  Status: Check failed ({type(e).__name__}: {e})")

    print("\n" + "=" * 40)
    if core_ok:
        print("Core dependencies OK. Real backends are optional.")
    else:
        print("Some core dependencies missing. Run 'uv sync --extra test' for basic setup.")
        print("Run 'uv sync --extra real' for full setup with real backends.")

    if strict and (not core_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
