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
    
    # Backward compatibility
    if "--streaming" in args:
        os.environ["KUCHIKAE_STREAMING_STT"] = "1"
        from kuchikae.web import serve
        serve()
        return
    
    print(f"Unknown command: {args[0]}")
    print("Run 'kuchikae --help' for usage.")
    sys.exit(1)


def print_help() -> None:
    port = os.environ.get("GRADIO_SERVER_PORT", "7860")
    print("""Kuchikae - Prompt-conditioned speech transformation

Usage:
  kuchikae serve [options]    Start the web server
  kuchikae doctor             Check backend availability
  kuchikae --help             Show this help

Options:
  --dummy         Use dummy backends for smoke testing
  --real          Use real backends (requires models)
  --streaming     Enable streaming STT for push-to-talk
  --port PORT     Server port (default: 7860)

Examples:
  kuchikae serve --dummy           # Smoke test with dummy backends
  kuchikae serve --real            # Use real STT/TTS backends
  kuchikae serve --streaming       # Enable streaming STT
  kuchikae serve --port 8080       # Use custom port
  kuchikae doctor                  # Check backend availability""")

    print(f"\nWeb UI will be available at http://127.0.0.1:{port}")


def cmd_serve(args: list[str]) -> None:
    dummy = "--dummy" in args
    real = "--real" in args
    streaming = "--streaming" in args
    
    port = None
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                print(f"Invalid port: {args[i + 1]}")
                sys.exit(1)
    
    if real and dummy:
        print("Cannot use both --real and --dummy.")
        sys.exit(1)
    
    from kuchikae.web import serve
    serve(dummy=dummy, real=real, streaming=streaming, port=port)


def cmd_doctor(args: list[str]) -> None:
    strict = "--strict" in args
    
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
        "OPENVOICE_READY",
        "KUCHIKAE_OPENVOICE_PATH",
    ]
    for var in env_vars:
        value = os.environ.get(var, "(not set)")
        if value and len(value) > 30:
            value = value[:30] + "..."
        print(f"  {var}: {value}")
    
    # Check dependencies
    print("\nDependencies:")
    deps = [
        ("gradio", "gradio"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("faster_whisper", "faster-whisper"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("irodori_tts", "irodori-tts"),
        ("httpx", "httpx"),
    ]
    
    all_ok = True
    for module_name, package_name in deps:
        try:
            __import__(module_name)
            print(f"  {package_name}: OK")
        except ImportError:
            print(f"  {package_name}: NOT INSTALLED")
            all_ok = False
    
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
    
    print("\n" + "=" * 40)
    if all_ok:
        print("All core dependencies installed.")
    else:
        print("Some dependencies missing. Run 'uv sync --extra real' for full setup.")
    
    if strict and not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
