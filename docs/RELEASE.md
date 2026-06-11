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
  __init__.py      # Public API exports
  cli.py           # CLI entry point (uv run kuchikae)
  web.py           # Web server entry
  pipeline.py      # Pipeline orchestration
  audio.py         # Audio chunking scaffold
  stt.py           # Speech-to-text backends
  text_transform.py# Text transform backends
  voice_output.py  # Voice output backends
  types.py         # Data types
  ui.py            # Gradio view layer
  logging.py       # Shared logging
  prompts/         # Package-data prompt files
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
