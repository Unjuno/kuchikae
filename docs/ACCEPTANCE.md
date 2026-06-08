# Kuchikae v0.1 Acceptance Criteria

This document defines how to accept or reject the initial implementation.

## 1. Required command flow

The implementation is accepted only if the following command flow is valid:

```bash
nix develop
uv sync
uv run pytest
uv run python app.py
```

If the implementation requires global pip, Homebrew-only setup, manually activated Python environments, or undocumented external models, reject it.

## 2. Product-level acceptance

The implementation must match this product shape:

```text
Audio input
  -> VoiceContext
  -> STT
  -> TextTransformPrompt-driven text transform
  -> VoiceOutputPrompt + VoiceContext-driven audio output
```

The implementation is accepted if:

- The main UI has an audio input.
- The main UI has a free-form Text Transform Prompt input.
- The main UI has a free-form Voice Output Prompt input.
- The main UI returns source transcript.
- The main UI returns transformed text.
- The main UI returns output audio.
- The main UI returns voice context readiness.
- The main UI returns latency report.

## 3. Immediate rejection conditions

Reject the implementation immediately if any of these are true:

1. The primary UI is a fixed style dropdown.
2. The app is framed as a polite-only rewriting app.
3. The app uses `polite/casual/rpg` choices as the main control instead of free-form prompts.
4. `VoiceOutputBackend.synthesize()` does not require `VoiceContext`.
5. `VoiceOutputBackend.synthesize()` does not require `VoiceOutputPrompt`.
6. The pipeline directly calls generic TTS without the `VoiceOutputBackend` abstraction.
7. Heavy ML packages are added in the first scaffold.
8. External model repositories are copied into this repository.
9. Generated outputs, model files, runs, or caches are tracked by Git.
10. Implementation introduces translation, streaming, database, authentication, or sharing.

## 4. Architecture acceptance

### 4.1 Required modules

The following files must exist:

```text
app.py
flake.nix
pyproject.toml
README.md
.gitignore
kuchikae/types.py
kuchikae/audio_cache.py
kuchikae/voice_context.py
kuchikae/stt.py
kuchikae/text_transform.py
kuchikae/voice_output.py
kuchikae/pipeline.py
kuchikae/timing.py
```

### 4.2 Required tests

The following test files must exist:

```text
tests/test_audio_cache.py
tests/test_voice_context.py
tests/test_text_transform_dummy.py
tests/test_voice_output_dummy.py
tests/test_pipeline_dummy.py
```

### 4.3 Required prompt files

```text
prompts/text_transform_default.txt
prompts/voice_output_default.txt
```

## 5. Data model acceptance

`kuchikae/types.py` must define:

- `ProsodyProfile`
- `VoiceContext`
- `TextTransformPrompt`
- `VoiceOutputPrompt`
- `LatencyReport`
- `PipelineResult`

Reject if:

- `TextTransformPrompt` is replaced by a fixed enum style.
- `VoiceOutputPrompt` is missing.
- `VoiceContext` is optional in the voice output path.
- `PipelineResult` uses only `styled_text` and fixed `style` semantics instead of `transformed_text` and prompt fields.

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

Reject if it uses only:

```python
def transform(self, text: str, style: str) -> str:
    ...
```

as the primary interface.

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

Reject if `voice_context` or `prompt` is optional or missing.

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

The Gradio app must include:

Inputs:

- audio input
- text transform prompt textbox
- voice output prompt textbox

Outputs:

- source transcript
- transformed text
- output audio
- voice context status
- latency report

Reject if the main UI uses a style dropdown as the primary control.

Prompt presets may be added later, but they must not replace the free-form prompt inputs.

## 9. Dependency acceptance

Initial dependencies may include:

- gradio
- numpy
- soundfile
- pyyaml
- pytest

Initial Nix dev shell may include:

- python311
- uv
- ffmpeg
- sox
- libsndfile
- portaudio
- git

Reject if first scaffold adds:

- torch
- transformers
- faster-whisper
- OpenVoice
- GPT-SoVITS
- CosyVoice
- F5-TTS
- RVC
- any heavy model dependency

These can be added later through separate explicit backend integration work.

## 10. Git hygiene acceptance

`.gitignore` must exclude:

```gitignore
.venv/
__pycache__/
*.pyc
outputs/*
!outputs/.gitkeep
runs/
models/
.cache/
.DS_Store
```

Reject if model weights, generated audio, large caches, or run directories are committed.

## 11. Latency reporting acceptance

`PipelineResult` must include a `LatencyReport` with:

- `stt_seconds`
- `text_transform_seconds`
- `voice_output_seconds`
- `total_seconds`

All values must be non-negative floats.

## 12. Review checklist

Before accepting Claude's implementation, check:

- [ ] Does `nix develop` enter the dev shell?
- [ ] Does `uv sync` complete?
- [ ] Does `uv run pytest` pass?
- [ ] Does `uv run python app.py` start the Gradio app?
- [ ] Is the main UI prompt-based rather than dropdown-based?
- [ ] Is `VoiceContext` required by voice output?
- [ ] Is `VoiceOutputPrompt` required by voice output?
- [ ] Does dummy voice output create a real WAV file?
- [ ] Are heavy model dependencies absent?
- [ ] Is the repository clean of outputs/models/runs?

## 13. Acceptance verdict format

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
