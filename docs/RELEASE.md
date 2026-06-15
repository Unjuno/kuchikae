# Release

## Build
- **package via uv build** - Use `uv build` to create wheel and sdist

## Install from wheel
- **install locally via uv tool install --force .** - Use `uv tool install --force .` for local development
- **run via kuchikae** - The CLI entry point is `kuchikae`

## GitHub Release
- **GitHub Release should attach wheel and sdist** - The dist/ directory contains both wheel and sdist
- **PyPI can come later** - Package may be ready for PyPI submission after testing

## Package structure
```
kuchikae/
  __init__.py           # Public API exports
  cli.py                # CLI entry point (uv run kuchikae)
  web.py                # Web server entry
  logging.py            # Shared logging
  models.py             # Model download / status
  quality.py            # Output quality metrics
  counting_backends.py  # Counting test doubles
  domain/
    __init__.py
    types.py            # Core data types (VoiceContext, PipelineResult, etc.)
    stt.py              # STTBackend interface + Dummy + Segmented
    text_transform.py   # TextTransformBackend + Ollama/GPT/Dummy/Rule
    voice_output.py     # VoiceOutputBackend interface + Dummy
    voice_prompt.py     # Voice prompt generation from emotion
    voice_style.py      # Voice style detection + fusion
    audio.py            # Audio chunking + linear_resample + torch_module
    audio_emotion.py    # Audio emotion detection
    audio_cache.py      # AudioCache + VoiceContextExtractor
    audio_key.py        # AudioKey helpers
    audio_stream.py     # Streaming audio types
    diagnostics.py      # Diagnostic event recorder
    error_hints.py      # Error message hints
    events.py           # DiagnosticEvent, EventLevel
    metrics.py          # LatencyLogger, counters
    processing_cache.py # ProcessingCache
    timing.py           # Timing utilities
  backends/
    __init__.py
    stt.py              # FasterWhisperSTTBackend
    stt_ct2.py          # CTranslate2 STT backend
    stt_nemo.py         # NeMo STT backend
    stt_transformers.py # Transformers Hubert backend
    stt_transformers_whisper.py  # Transformers Whisper backend
    voice_output.py     # IrodoriTTS + OpenVoice backends
  pipeline/
    __init__.py
    pipeline.py         # KuchikaePipeline orchestration
    audio_validation.py # Audio validation helpers
  ui/
    __init__.py
    app.py              # Gradio component tree
    handlers.py         # Event handlers
    templates.py        # Template data
    css.py              # Styles
    js.py               # Client-side JS
  prompts/
    text_transform_default.txt
    voice_output_default.txt
```

## Model weights
- **model weights must not be bundled in wheel or release assets** - All model files are external or via package managers
- **release artifacts belong in dist/, but dist/ should not be committed** - Use .gitignore to prevent accidental commits

## Dependencies
See `pyproject.toml`. Core deps are lightweight (gradio, numpy, soundfile).
Heavy deps (torch, faster-whisper, irodori-tts) are optional at import time.

## Versioning
`0.1.x` — prototype phase. Breaking changes expected.
