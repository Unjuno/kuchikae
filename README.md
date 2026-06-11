# Kuchikae

Kuchikae is a prompt-conditioned, voice-conditioned speech transformation prototype.

Core idea:

> Speak once. Kuchikae says it back in your voice, following your prompt.

The app takes a short Japanese speech input, uses that audio both as the utterance to transcribe and as reference audio for `VoiceContext`, transforms the transcript according to a free-form `TextTransformPrompt`, and generates output speech through `VoiceOutputBackend` using `VoiceContext` and `VoiceOutputPrompt`.

This repository is intentionally specification-first. Implementation must follow the documents under `docs/`.

## Fixed v0.1 scope

v0.1 is not a fixed-style dropdown app. The main controls are free-form prompts:

- `TextTransformPrompt`: controls how the transcript is rewritten.
- `VoiceOutputPrompt`: controls how the generated speech should sound.

v0.1 must start with dummy backends and a Nix + uv scaffold. Real STT and real voice-conditioned output backends are added only after the architecture is in place.

## Implemented features

- **Prompt-based text transformation** with few-shot templates (polite, casual, summarize)
- **Voice-conditioned output** via Irodori-TTS (or OpenVoice)
- **Caching layer** (`ProcessingCache`) avoids redundant STT and VoiceContext extraction
- **Streaming pipeline** (`process_stream_live`) for push-to-talk UX with partial transcripts
- **Segmented STT** for long audio via `FixedWindowSegmenter`
- **Fast rule-based text transform** as default (no LLM required)
- **Progressive Gradio UI** shows transcript → transformed text → audio

## Required reading for implementation

1. [`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md) — clone, Nix, and uv workflow.
2. [`docs/SPEC.md`](docs/SPEC.md) — fixed product and user-facing specification.
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — backend interfaces and data flow.
4. [`docs/CLAUDE_IMPLEMENTATION_CONTRACT.md`](docs/CLAUDE_IMPLEMENTATION_CONTRACT.md) — exact implementation contract for Claude or any coding agent.
5. [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) — acceptance and rejection criteria.

## Non-goals for v0.1

Do not implement:

- fixed style dropdown as the primary UI
- translation
- user accounts
- database
- sharing
- real voice cloning inside the initial scaffold
- heavy model dependencies in the first commit

## Local setup and target development commands

Clone the repository first:

```bash
git clone git@github.com:Unjuno/kuchikae.git
cd kuchikae
```

If SSH is not configured, use HTTPS:

```bash
git clone https://github.com/Unjuno/kuchikae.git
cd kuchikae
```

The implementation must then support:

```bash
nix develop
uv sync
uv run pytest
uv run python app.py
```

Do not use global Python, global pip, or undeclared Homebrew-only dependencies. Nix owns the system development shell; uv owns Python package resolution.

## Configuration

Backend selection via `create_pipeline()` config:

```python
# Default: fast rule-based text transform + Irodori-TTS voice
create_pipeline()

# Streaming STT for push-to-talk (partial transcripts)
create_pipeline({"streaming_stt": True})

# Segmented STT for long audio
create_pipeline({"segmented_stt": True})

# Ollama LLM text transform (requires Ollama server)
create_pipeline({"text_transform_backend": "ollama", "text_transform_model": "qwen3:8b"})
```
