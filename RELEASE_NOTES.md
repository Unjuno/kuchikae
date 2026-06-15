# Kuchikae v0.1.1

**Release date:** 2026-06-15

## Requirements

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) — fast Python package manager
- (optional) [Ollama](https://ollama.com/) for LLM text transform
- (optional) ~6 GB disk for model downloads

## Installation

### From GitHub release (zip/tar.gz)

```bash
# Download and extract
curl -sL https://github.com/Unjuno/kuchikae/archive/refs/tags/v0.1.1.tar.gz | tar xz
cd kuchikae-0.1.1

# Smoke test (no models required)
make run
# → http://127.0.0.1:7860

# Real backends (downloads models automatically)
make run-real
```

### From git clone

```bash
git clone https://github.com/Unjuno/kuchikae.git
cd kuchikae
make run           # smoke test
make run-real      # real backends
```

## Quick start commands

| Command | What it does |
|---------|-------------|
| `make run` | Install deps + start dummy server |
| `make run-real` | Install real deps + download models + start |
| `make run-streaming` | Real deps + streaming STT |
| `make doctor` | Check backend/model status |
| `make test` | Run lightweight tests |
| `make check-all` | Full lint + typecheck + audit + tests |

## Manual usage (without Makefile)

```bash
# Smoke test
uv sync
uv run kuchikae serve --dummy

# Real backends
uv sync --extra real
uv run kuchikae setup-models --all
uv run kuchikae serve --real
```

## Changes since v0.1.0

### 🔍 Full code audit (6 rounds)

- Bug fixes: Irodori-TTS caption passing, audio emotion init, voice context cache
- Dead code removal: VoicePromptResolver (261 lines), _shift_speed, repair_model_by_name
- Code dedup: _linear_resample and _torch_module extracted to kuchikae/domain/audio.py
- Temp file hygiene, weak assertions hardened, silent exception fixed

### ✅ Lint, type check & security (zero issues)

- `ruff`: 0 errors (fixed 28 files)
- `mypy`: 0 errors (fixed 54→0 across 12 categories)
- `pip-audit`: 0 vulnerabilities
- `pytest`: 449 passed, 13 skipped (1 pre-existing flaky)

### 🎤 Voice evaluation framework

- `evals/run_voice_eval.py` with `--mode tts-only|pipeline`
- 42 eval cases, baseline results, `docs/VOICE_EVAL_PLAN.md`

### 🎛️ Simplified startup

- `Makefile` with `make run`, `make run-real`, `make run-streaming`

### 📦 Package structure

```
kuchikae/
  domain/      # Core interfaces & types
  backends/    # STT, TTS, emotion implementations
  pipeline/    # KuchikaePipeline orchestration
  ui/          # Gradio component tree
```

## Known limitations

- Prototype phase — breaking changes may occur
- Dummy mode is for smoke testing only
- Real STT/TTS quality depends on local models and hardware
- OpenVoice integration is experimental
- Exact voice cloning is not guaranteed

## License

MIT. See `LICENSE` file.
