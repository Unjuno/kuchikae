# Kuchikae v0.1 Acceptance Criteria

This document defines how to accept or reject the v0.1 implementation.

## 1. Required command flow

The implementation is accepted only if the following command flow is valid from a fresh local clone:

```bash
git clone https://github.com/Unjuno/kuchikae.git
cd kuchikae
uv sync --extra test
uv run pytest -q -m "not slow and not e2e"
uv run python -m compileall kuchikae
uv run kuchikae --help
uv run kuchikae doctor
uv run kuchikae serve --dummy
```

If the implementation requires global pip, Homebrew-only setup, manually activated Python environments, or undocumented external models, reject it.

## 2. Product-level acceptance

The implementation must match this product shape:

```text
Audio input
  -> VoiceContext
  -> STT
  -> Text transform (template or free-form prompt)
  -> VoiceOutputPrompt (internally generated from emotion analysis)
  -> VoiceOutputBackend + VoiceContext -> output audio
```

The implementation is accepted if:

- The main UI has an audio input (microphone or upload).
- The main UI has a template dropdown with 通常/正式/実験/実験強/カスタム categories.
- The main UI has a free-form prompt textbox (Normal mode).
- The main UI has a PTT button (Simple mode).
- The main UI returns source transcript.
- The main UI returns transformed text.
- The main UI returns output audio.
- The main UI displays voice impression label (声の印象).
- Experimental templates display a safety warning.

## 3. Immediate rejection conditions

Reject the implementation immediately if any of these are true:

1. `VoiceOutputPrompt` is exposed as a user-facing textbox in the UI.
2. Heavy ML packages are required base dependencies (not optional).
3. External model repositories are copied into this repository.
4. Generated outputs, model files, runs, or caches are tracked by Git.
5. The app has no dummy mode for smoke testing.
6. Experimental templates are hidden from the UI.
7. The safety warning is removed from experimental templates.

## 4. Architecture acceptance

### 4.1 Required modules

The following files must exist:

```text
pyproject.toml
README.md
LICENSE
.gitignore
kuchikae/__init__.py
kuchikae/cli.py
kuchikae/ui/app.py
kuchikae/ui/css.py
kuchikae/ui/js.py
kuchikae/ui/handlers.py
kuchikae/ui/templates.py
kuchikae/domain/types.py
kuchikae/domain/text_transform.py
kuchikae/domain/voice_output.py
kuchikae/domain/voice_prompt.py
kuchikae/pipeline/pipeline.py
```

### 4.2 Required tests

Test files must include coverage for:

- UI structure (create_app, template presence, warning presence)
- Template keys (all categories present)
- Text transform validation (echo detection, CJK detection, meta prefix rejection)
- Voice prompt generation (emotion-to-prompt mapping)
- Pipeline dummy mode

## 5. Data model acceptance

`kuchikae/domain/types.py` must define:

- `TextTransformPrompt`
- `VoiceOutputPrompt`
- `VoiceContext`
- `PipelineResult`

Reject if:

- `VoiceOutputPrompt` is missing.
- `VoiceOutputPrompt` is required as a user-facing input.

## 6. Backend interface acceptance

### 6.1 STTBackend

Must expose:

```python
def transcribe(self, audio_path: str) -> str:
    ...
```

### 6.2 TextTransformBackend

Must expose:

```python
def transform(self, text: str, prompt: TextTransformPrompt) -> str:
    ...
```

### 6.3 VoiceOutputBackend

Must expose:

```python
def synthesize(
    self,
    text: str,
    voice_context: VoiceContext,
    prompt: VoiceOutputPrompt,
) -> str:
    ...
```

## 7. Dummy backend acceptance

The first implementation must run without real models.

Accepted dummy behavior:

- `DummySTTBackend` returns a fixed Japanese transcript.
- `DummyTextTransformBackend` returns a non-empty transformed sentence.
- `DummyVoiceOutputBackend` writes a valid silent WAV file.

Rejected dummy behavior:

- dummy voice output returns a fake path without writing a valid file
- dummy pipeline requires API keys
- dummy pipeline requires downloaded model weights

## 8. UI acceptance

### 8.1 Normal mode (通常)

The Gradio app must include:

Inputs:

- audio input (microphone + upload)
- template dropdown
- text transform prompt textbox
- voice style radio (auto/natural/calm/bright/slow_clear)
- run button ("言い直す")

Outputs:

- source transcript
- transformed text
- output audio
- voice impression label (声の印象)

### 8.2 Simple mode (簡易)

The Gradio app must include:

- template dropdown
- PTT button ("押して話す")
- hidden audio component (CSS offscreen hidden, visible in DOM)
- source transcript
- transformed text
- output audio

Reject if:

- The PTT button uses `transform: scale()` in recording state.
- The hidden audio component uses `display: none`.
- Simple mode layout shifts during recording.

## 9. Dependency acceptance

Initial dependencies may include:

- gradio
- numpy
- soundfile
- pyyaml
- httpx
- pytest (test extra)

Heavy dependencies must be optional extras:

- faster-whisper (stt extra)
- torch, torchaudio, irodori-tts, silentcipher (tts extra)

Reject if first scaffold adds heavy dependencies as required base dependencies.

## 10. Git hygiene acceptance

`.gitignore` must exclude:

```gitignore
.venv/
__pycache__/
*.pyc
*.egg-info/
dist/
build/
models/
checkpoints/
.cache/
*.safetensors
*.bin
*.onnx
*.pt
*.pth
*.ckpt
outputs/*.wav
logs/
artifacts/
```

Reject if model weights, generated audio, large caches, or run directories are committed.

## 11. Review checklist

Before accepting the implementation, check:

- [ ] Does a fresh `git clone` + `uv sync --extra test` work?
- [ ] Does `uv run pytest -q -m "not slow and not e2e"` pass?
- [ ] Does `uv run python -m compileall kuchikae` succeed?
- [ ] Does `uv run kuchikae --help` show help?
- [ ] Does `uv run kuchikae doctor` show backend status?
- [ ] Does `uv run kuchikae serve --dummy` start the Gradio app?
- [ ] Is the main UI template-based with prompt override?
- [ ] Are experimental templates visible with safety warnings?
- [ ] Is `VoiceOutputPrompt` internal (not user-facing)?
- [ ] Does dummy voice output create a real WAV file?
- [ ] Are heavy model dependencies absent from base install?
- [ ] Is the repository clean of outputs/models/runs?
- [ ] Is LICENSE present with correct copyright?

## 12. Acceptance verdict format

Use this format after reviewing an implementation:

```text
Verdict: PASS / FAIL

Blocking issues:
- ...

Non-blocking issues:
- ...

Required fixes:
- ...
```
