# Kuchikae v0.1.0 Release Notes

**Release date:** 2026-06-13

## Highlights

Kuchikae v0.1 is a prompt-conditioned, voice-conditioned speech transformation prototype for Japanese.

> Speak once. Kuchikae says it back in your voice, following your prompt.

## What's new

### CLI interface

- `kuchikae serve` — start the web server
- `kuchikae doctor` — check backend availability
- `kuchikae serve --dummy` — smoke test with dummy backends
- `kuchikae serve --real` — use real STT/TTS backends
- `kuchikae serve --real --streaming` — enable faster-whisper streaming STT

### Voice style detection

- **Text-based rules**: detects urgent, apologetic, warm, or serious tone from transcript
- **Audio emotion**: optional parallel analysis of speaker audio
- **Fusion**: combines text style and optional audio emotion with confidence weighting
- **VoiceOutputPrompt** can be generated automatically from detected style

### Pipeline architecture

- `VoicePromptResolver` handles voice style detection and fusion
- Text transform validation and fallback are handled at the pipeline layer
- Optional audio emotion detection can run in parallel with STT
- Streaming STT is available for real-backend push-to-talk experiments

### Backends

- **STT**: dummy smoke backend, optional faster-whisper, streaming, segmented
- **Text transform**: prompted rule, rule-based, optional Ollama LLM, optional GPT backend
- **Voice output**: dummy smoke backend, optional Irodori-TTS, experimental OpenVoice

## Quick start

### Smoke test (no models required)

```bash
uv sync --extra test
uv run kuchikae serve --dummy
```

Opens Gradio at `http://127.0.0.1:7860`. Dummy mode is for smoke testing only; it does not evaluate real STT or voice quality.

### Real backends

```bash
uv sync --extra real
uv run kuchikae serve --real
```

Requires:

- faster-whisper models
- Ollama running with a model
- Irodori-TTS models

### Streaming STT

```bash
uv sync --extra real
uv run kuchikae serve --real --streaming
```

Streaming STT is a real-backend mode.

### Check backends

```bash
uv run kuchikae doctor
```

Shows installed backends, Ollama status, and selected environment variables.

## Validation

Validate locally before tagging the release:

```bash
uv sync --extra test
uv run pytest -q -m "not slow and not e2e"
uv run python -m compileall kuchikae
uv run kuchikae doctor
uv run kuchikae serve --dummy
```

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KUCHIKAE_STT_BACKEND` | STT backend | `dummy` for default smoke mode, `faster_whisper` with `--real` |
| `KUCHIKAE_TEXT_BACKEND` | Text transform backend | `prompted_rule` |
| `KUCHIKAE_VOICE_BACKEND` | Voice output backend | `dummy` for default smoke mode, `irodori` with `--real` |
| `KUCHIKAE_STREAMING_STT` | Request streaming STT | `0` |
| `KUCHIKAE_TEXT_MODEL` | Ollama model name | backend default |
| `KUCHIKAE_ALLOW_DUMMY_BACKENDS` | Allow dummy fallback | `1` for default smoke mode, `0` with `--real` |

## Documentation

- [`README.md`](README.md) — quick start and CLI usage
- [`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md) — setup instructions
- [`docs/SPEC.md`](docs/SPEC.md) — product specification
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — backend interfaces

## Known limitations

- This is a prototype.
- Dummy mode is only for smoke testing.
- Exact voice cloning is not guaranteed.
- Real STT/TTS quality depends on local models and hardware.
- Streaming STT requires real backend setup.
- OpenVoice integration is experimental.

## Credits

Built with optional integrations around:

- [Gradio](https://gradio.app/) — web UI
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — STT
- [Ollama](https://ollama.com/) — LLM inference
- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) — voice synthesis
