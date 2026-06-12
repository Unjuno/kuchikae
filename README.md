# Kuchikae

Kuchikae is a prompt-conditioned, voice-conditioned speech transformation prototype.

Core idea:

> Speak once. Kuchikae says it back in your voice, following your prompt.

The app takes a short Japanese speech input, uses that audio both as the utterance to transcribe and as reference audio for `VoiceContext`, transforms the transcript according to a free-form `TextTransformPrompt`, and generates output speech through `VoiceOutputBackend` using `VoiceContext` and `VoiceOutputPrompt`.

## Quick start

### Smoke test (dummy backends)

```bash
uv run python -m kuchikae.cli serve --dummy
```

Opens Gradio at `http://127.0.0.1:7860`. Record audio → get dummy output. No models required.

### Real backends

```bash
uv sync --extra real
uv run python -m kuchikae.cli serve --real
```

Requires: faster-whisper models, Ollama running, Irodori-TTS models.

### Doctor (check backend availability)

```bash
uv run python -m kuchikae.cli doctor
```

Shows installed backends, Ollama status, environment variables, and missing dependencies.

## CLI usage

```
kuchikae serve [options]    Start the web server
kuchikae doctor             Check backend availability
kuchikae --help             Show help

Options:
  --dummy         Use dummy backends for smoke testing
  --real          Use real backends (requires models)
  --streaming     Enable streaming STT for push-to-talk
  --port PORT     Server port (default: 7860)
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
uv run python -m kuchikae.cli serve --dummy      # smoke test
uv run python -m kuchikae.cli serve --real        # real backends
uv run python -m kuchikae.cli serve --streaming   # streaming STT

# Environment variables
export KUCHIKAE_STT_BACKEND=faster_whisper
export KUCHIKAE_TEXT_BACKEND=ollama
export KUCHIKAE_VOICE_BACKEND=irodori
export KUCHIKAE_STREAMING_STT=1

# Ollama LLM text transform
export KUCHIKAE_TEXT_BACKEND=ollama
export KUCHIKAE_TEXT_MODEL=qwen3:8b
```

## Development

```bash
nix develop    # enter dev shell
uv sync        # install dependencies
uv run pytest  # run tests
uv run python -m kuchikae.cli doctor  # check backends
uv run python -m kuchikae.cli serve --dummy  # smoke test
```

## Non-goals for v0.1

Do not implement:

- fixed style dropdown as the primary UI
- translation
- user accounts
- database
- sharing
- real voice cloning inside the initial scaffold
- heavy model dependencies in the first commit

## Documentation

1. [`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md) — clone, Nix, and uv workflow.
2. [`docs/SPEC.md`](docs/SPEC.md) — fixed product and user-facing specification.
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — backend interfaces and data flow.
4. [`docs/CLAUDE_IMPLEMENTATION_CONTRACT.md`](docs/CLAUDE_IMPLEMENTATION_CONTRACT.md) — exact implementation contract.
5. [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) — acceptance and rejection criteria.
