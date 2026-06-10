# Release

## Build
```bash
uv build
```

## Install from wheel
```bash
uv pip install dist/kuchikae-0.1.0-py3-none-any.whl
```

## Run
```bash
kuchikae          # CLI entry point (starts web server)
kuchikae web      # same
```

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
```

## Dependencies
See `pyproject.toml`. Core deps are lightweight (gradio, numpy, soundfile).
Heavy deps (torch, faster-whisper, irodori-tts) are optional at import time.

## Versioning
`0.1.x` — prototype phase. Breaking changes expected.
