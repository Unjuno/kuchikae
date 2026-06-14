# Kuchikae v0.1.0 Release Notes

## Summary

Kuchikae v0.1.0 is the initial public release of a prompt-conditioned, voice-conditioned Japanese speech transformation prototype. Speak once, and Kuchikae says it back in your voice, following your prompt.

## What's included

### Two modes

- **Normal mode (通常)**: Record audio, select a template or write a free-form prompt, click "言い直す".
- **Simple mode (簡易)**: Select a template, press and hold the PTT button, speak, release for automatic conversion.

### 33 built-in templates

- 12 original templates (natural, polite, casual, short, strong, calm, announcer, movie trailer, AI assistant, butler, demon king, late-night radio)
- 5 official additional templates (teacher, friend, news caster, sales, poetic)
- 8 experimental templates (`実験:`) — Kansai dialect, gyaru, baby, samurai, sharp-tongued, sarcastic, foreign, specific character
- 8 strong experimental templates (`実験強:`) — stronger variants for red-team and demo purposes
- Custom template for free-form prompt input

### Voice transformation pipeline

- STT (FasterWhisper or dummy)
- Text transformation (Ollama LLM, rule-based, or dummy)
- Voice output (Irodori-TTS or dummy)
- Audio emotion analysis for automatic voice prompt generation
- Caching layer to avoid redundant processing

### Safety features

- Experimental templates display a safety warning in the UI.
- Text validation rejects meta-output, prompt echoes, CJK-only Chinese, and identity-closed loops.
- Voice prompt is internally generated from emotion analysis (not user-editable).

### CLI

```
kuchikae serve [options]    Start the web server
kuchikae doctor             Check backend availability
kuchikae --help             Show help
```

Options:
- `--dummy` — dummy backends for smoke testing (no models required)
- `--real` — real backends (requires Ollama, FasterWhisper, Irodori-TTS)
- `--streaming` — streaming STT with real backends
- `--port PORT` — server port (default: 7860)

## Safety

This tool is for research and prototype purposes only. Users must:

- Use only their own audio or audio with explicit speaker permission.
- Not use the tool for impersonation, fraud, harassment, threats, political deception, or non-consensual voice imitation.

Experimental templates are for red-team testing, demo purposes, and safety evaluation only.

## Known limitations

- **Exact voice cloning is not guaranteed.** Output voice quality depends on the TTS backend and input audio quality.
- **Real backend is environment-dependent.** Ollama, FasterWhisper models, and Irodori-TTS models must be installed separately. The `--dummy` mode is provided for smoke testing without any model dependencies.
- **Japanese only.** Input language is Japanese. Other languages are not supported.
- **Short utterances only.** Recommended input length is 3-10 seconds. Maximum is 15 seconds.
- **LLM output quality varies.** The Ollama text transformation may occasionally produce Chinese text, meta-output, or refuse to transform. The validation layer catches most cases but is not perfect.

## Smoke test commands

```bash
# Install dependencies
uv sync --extra test

# Run tests
uv run pytest -q -m "not slow and not e2e"

# Compile check
uv run python -m compileall kuchikae

# CLI check
uv run kuchikae --help
uv run kuchikae doctor

# Build package
uv build

# Start dummy server
uv run kuchikae serve --dummy
```

## Real backend setup (optional)

```bash
# Install real backend dependencies
uv sync --extra real

# Start Ollama
ollama serve
ollama pull qwen2.5:7b-instruct
# Run `ollama list` to check available models on your system.

# Start Kuchikae with real backends
uv run kuchikae serve --real
```

## Eval results note

Previous eval runs used `qwen2.5-coder:7b` (a coder model). These results are not representative of production quality and should not be used for release judgment. The recommended text transform model is now `qwen2.5:7b-instruct`. Run `ollama list` to verify available models on your system.

## License

MIT License. Copyright (c) 2026 Unjuno.
