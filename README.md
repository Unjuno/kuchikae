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

## Required reading for implementation

1. [`docs/SPEC.md`](docs/SPEC.md) — fixed product and user-facing specification.
2. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — backend interfaces and data flow.
3. [`docs/CLAUDE_IMPLEMENTATION_CONTRACT.md`](docs/CLAUDE_IMPLEMENTATION_CONTRACT.md) — exact implementation contract for Claude or any coding agent.
4. [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) — acceptance and rejection criteria.

## Non-goals for v0.1

Do not implement:

- fixed style dropdown as the primary UI
- translation
- streaming
- user accounts
- database
- sharing
- real voice cloning inside the initial scaffold
- heavy model dependencies in the first commit

## Target development commands

The implementation must eventually support:

```bash
nix develop
uv sync
uv run pytest
uv run python app.py
```
