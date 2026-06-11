# Security Review

## Attack surface
- Uploaded audio files
- Prompt text (user-supplied instruction to LLM)
- Output audio paths

## Controls

### Audio file validation
- **validate uploaded audio extension** - only `.wav` allowed (rejected otherwise)
- **validate file size** - max 25 MB per file
- **validate audio duration** - max 30 seconds per file
- **do not trust uploaded filenames** - `normalize_audio_path()` resolves through Gradio's temp layer; raw filename is never trusted
- **write temp/chunk files under a controlled directory** - temporary chunk files use temp directory under project control
- **clean temporary chunk files** - all temporary chunk files are cleaned up

### Prompt safety
- **do not upload user audio to third-party services unless explicitly configured** - prompt text is passed to the local LLM (Ollama) and never sent to an external API unless the user explicitly configures `GPTTextTransformBackend` with `OPENAI_API_KEY`.
- **dummy mode must require no API keys** - Dummy mode requires no API keys.

### Output path safety
- **never allow user-controlled output paths** - Output audio is always written to `outputs/` under the project root.
- The output filename is hardcoded per backend (`dummy.wav`, `irodori_output.wav`, `openvoice_output.wav`).
- User-controlled strings never appear in output paths.

### Data retention
- Uploaded audio files live in Gradio's temp directory and are cleaned up by Gradio.
- The `outputs/` directory is gitignored (`.gitignore`).
- Pipeline caches (`_stt_cache`, `_voice_cache`) are in-memory only and lost on restart.

### Git hygiene
- **never commit generated/user audio, model weights, external repos, cache files, or secrets** - All heavy dependencies and model files are external or via package managers.

## Threat model
- **Local single-user desktop app.** No network service beyond localhost Gradio.
- **No authentication boundary.** Assumes the user controls the machine.
- **Ollama runs locally.** Prompt data never leaves the host.
- **PyPI supply chain.** Dependencies are pinned in `pyproject.toml` and `uv.lock`.
