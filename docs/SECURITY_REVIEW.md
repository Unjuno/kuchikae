# Security Review

## Attack surface
- Uploaded audio files
- Prompt text (user-supplied instruction to LLM)
- Output audio paths

## Controls

### Audio file validation
- Extension: only `.wav` allowed (rejected otherwise)
- Size: max 25 MB per file
- Duration: max 30 seconds per file
- Path: `normalize_audio_path()` resolves through Gradio's temp layer; raw filename is never trusted

### Prompt safety
- Prompt text is passed to the local LLM (Ollama) and never sent to an external API unless the user explicitly configures `GPTTextTransformBackend` with `OPENAI_API_KEY`.
- Dummy mode requires no API keys.

### Output path safety
- Output audio is always written to `outputs/` under the project root.
- The output filename is hardcoded per backend (`dummy.wav`, `irodori_output.wav`, `openvoice_output.wav`).
- User-controlled strings never appear in output paths.

### Data retention
- Uploaded audio files live in Gradio's temp directory and are cleaned up by Gradio.
- The `outputs/` directory is gitignored (`.gitignore`).
- Pipeline caches (`_stt_cache`, `_voice_cache`) are in-memory only and lost on restart.

## Threat model
- **Local single-user desktop app.** No network service beyond localhost Gradio.
- **No authentication boundary.** Assumes the user controls the machine.
- **Ollama runs locally.** Prompt data never leaves the host.
- **PyPI supply chain.** Dependencies are pinned in `pyproject.toml` and `uv.lock`.
