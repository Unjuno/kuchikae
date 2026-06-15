# Kuchikae

Kuchikae is a prompt-conditioned, voice-conditioned speech transformation prototype.

Core idea:

> Speak once. Kuchikae says it back in your voice, following your prompt.

The app takes a short Japanese speech input, uses that audio both as the utterance to transcribe and as reference audio for `VoiceContext`, transforms the transcript according to a free-form `TextTransformPrompt`, and generates output speech through `VoiceOutputBackend` using `VoiceContext` and `VoiceOutputPrompt`.

## Quick start

### Requirements
- Python 3.11
- `uv` (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- (optional) [Ollama](https://ollama.com/) for LLM text transform
- (optional) ~6 GB disk for model downloads

### Smoke test (no models required)

```bash
make run
```

Opens Gradio at `http://127.0.0.1:7860`. Record audio → get dummy output.

### Real backends (STT + TTS)

```bash
make run-real
```

This installs real dependencies, downloads models, and starts the server.

### Streaming STT

```bash
make run-streaming
```

### Doctor (check backend availability)

```bash
make doctor
```

Shows installed backends, Ollama status, environment variables, and missing dependencies.

### Manual steps (without Makefile)

```bash
# Install
uv sync

# Smoke test
uv run kuchikae serve --dummy

# Real backends
uv sync --extra real
uv run kuchikae setup-models --all
uv run kuchikae serve --real
```

Models are downloaded to the standard Hugging Face cache directory (`~/.cache/huggingface/`). They are not bundled in this repository.

### Repairing broken models

```bash
make doctor       # check status
uv run kuchikae doctor --fix
# or
uv run kuchikae setup-models --all --repair
```

## CLI usage

```
kuchikae serve [options]          Start the web server
kuchikae doctor [--fix]           Check backend availability
kuchikae setup-models [options]   Download required model weights
kuchikae --help                   Show help

Options:
  --dummy         Use dummy backends for smoke testing
  --real          Use real backends (requires models)
  --streaming     Enable streaming STT with --real
  --port PORT     Server port (default: 7860)

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
```

## Fixed v0.1 scope

v0.1 is not a fixed-style dropdown app. The main controls are free-form prompts:

- `TextTransformPrompt`: controls how the transcript is rewritten.
- `VoiceOutputPrompt`: controls how the generated speech should sound.

## Implemented features

- **Prompt-based text transformation** with few-shot templates (polite, casual, summarize)
- **Voice-conditioned output** via Irodori-TTS (or OpenVoice)
- **Caching layer** (`ProcessingCache`) avoids redundant STT and VoiceContext extraction
- **Streaming pipeline** (`process_stream_live`) for push-to-talk UX with partial transcripts
- **Segmented STT** for long audio via `FixedWindowSegmenter`
- **Fast rule-based text transform** as default (no LLM required)
- **Progressive Gradio UI** shows transcript → transformed text → audio
- **Voice style detection** from text rules, audio emotion, and LLM analysis
- **CLI interface** with serve, doctor, and mode flags

## Configuration

Backend selection via CLI flags or environment variables:

```bash
# CLI flags (recommended)
uv run kuchikae serve --dummy              # smoke test
uv run kuchikae serve --real               # real backends
uv run kuchikae serve --real --streaming   # real streaming STT

# Environment variables
export KUCHIKAE_STT_BACKEND=faster_whisper
export KUCHIKAE_TEXT_BACKEND=ollama
export KUCHIKAE_VOICE_BACKEND=irodori
export KUCHIKAE_STREAMING_STT=1

# Ollama LLM text transform
export KUCHIKAE_TEXT_BACKEND=ollama
export KUCHIKAE_TEXT_MODEL=qwen2.5:7b-instruct
# Run `ollama list` to check available models on your system.
```

## Development

```bash
nix develop    # enter dev shell
uv sync        # install dependencies
uv run pytest  # run tests
uv run kuchikae doctor  # check backends
uv run kuchikae serve --dummy  # smoke test
```

## Non-goals for v0.1

Do not implement:

- fixed style dropdown as the primary UI
- translation
- user accounts
- database
- sharing
- exact voice cloning guarantee
- heavy model dependencies as required base dependencies

## Safety and consent

Kuchikae is a research and prototype tool for exploring prompt-conditioned voice transformation.

**You must:**

- Use only your own audio, or audio for which you have explicit permission from the speaker.
- Use the tool in compliance with all applicable laws and regulations.

**Do not use Kuchikae for:**

- Impersonation, fraud, or phishing.
- Harassment, threats, or bullying.
- Political deception or disinformation campaigns.
- Non-consensual voice imitation or deepfake creation.

**Experimental templates** (`実験:`, `実験強:`) are provided for red-team testing, demo purposes, and safety evaluation. They are not intended for production use or abuse.

The developers assume no liability for misuse of this tool. Users are solely responsible for ensuring their use complies with ethical and legal standards.

## Documentation

1. [`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md) — clone, Nix, and uv workflow.
2. [`docs/SPEC.md`](docs/SPEC.md) — fixed product and user-facing specification.
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — backend interfaces and data flow.
4. [`docs/CLAUDE_IMPLEMENTATION_CONTRACT.md`](docs/CLAUDE_IMPLEMENTATION_CONTRACT.md) — exact implementation contract.
5. [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) — acceptance and rejection criteria.
