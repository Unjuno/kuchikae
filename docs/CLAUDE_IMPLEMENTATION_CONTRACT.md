# Claude Implementation Contract for Kuchikae v0.1

This document is the implementation contract for Claude or any coding agent.

Claude must implement exactly this scaffold. Claude must not redesign the product.

## 1. Objective

Create a Nix-managed Python project scaffold for Kuchikae v0.1.

The scaffold must run with dummy backends and preserve the final architecture:

```text
Audio input
  -> VoiceContext
  -> STT
  -> TextTransformPrompt-driven text transformation
  -> VoiceOutputPrompt + VoiceContext-driven audio output
```

## 2. Product constraints

Kuchikae is not a fixed-style dropdown app.

The primary controls are:

1. `TextTransformPrompt`
2. `VoiceOutputPrompt`

A fixed style dropdown is not allowed as the primary interface.

## 3. Required stack

Use:

- Nix flakes
- Python 3.11
- uv
- pytest
- Gradio
- numpy
- soundfile
- pyyaml
- ffmpeg in dev shell
- sox in dev shell
- libsndfile in dev shell
- portaudio in dev shell

Do not add heavy ML dependencies in the first scaffold.

## 4. Required repository tree

Create exactly this project shape:

```text
kuchikae/
  flake.nix
  pyproject.toml
  README.md
  .gitignore
  app.py

  kuchikae/
    __init__.py
    types.py
    audio_cache.py
    voice_context.py
    stt.py
    text_transform.py
    voice_output.py
    pipeline.py
    timing.py

  prompts/
    text_transform_default.txt
    voice_output_default.txt

  outputs/
    .gitkeep

  tests/
    test_audio_cache.py
    test_voice_context.py
    test_pipeline_dummy.py
    test_text_transform_dummy.py
    test_voice_output_dummy.py
```

## 5. `flake.nix` requirements

Create `flake.nix` with:

- `nixpkgs` from `nixos-unstable`
- `flake-utils`
- default dev shell
- packages:
  - `python311`
  - `uv`
  - `ffmpeg`
  - `sox`
  - `libsndfile`
  - `portaudio`
  - `git`

Shell hook must set:

```bash
export UV_PROJECT_ENVIRONMENT=.venv
export PYTHONPATH=$PWD
```

Shell hook must print:

```text
Kuchikae dev shell
Run: uv sync
Run: uv run pytest
Run: uv run python app.py
```

## 6. `pyproject.toml` requirements

Use:

```toml
[project]
name = "kuchikae"
version = "0.1.0"
description = "Prompt-conditioned voice-conditioned speech transformation prototype"
requires-python = ">=3.11,<3.12"
dependencies = [
  "gradio",
  "numpy",
  "soundfile",
  "pyyaml",
  "pytest",
]
```

Do not add:

- torch
- transformers
- faster-whisper
- OpenVoice
- GPT-SoVITS
- CosyVoice
- F5-TTS
- RVC

## 7. `.gitignore` requirements

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

## 8. Required dataclasses

Implement the dataclasses specified in `docs/ARCHITECTURE.md` exactly unless there is a clear Python syntax issue.

Required names:

- `ProsodyProfile`
- `VoiceContext`
- `TextTransformPrompt`
- `VoiceOutputPrompt`
- `LatencyReport`
- `PipelineResult`

Do not rename `TextTransformPrompt` to `Style`.
Do not rename `transformed_text` to `styled_text`.

## 9. Required backend interfaces

### 9.1 STTBackend

```python
class STTBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        ...
```

Required dummy:

```python
class DummySTTBackend(STTBackend):
    def transcribe(self, audio_path: str) -> str:
        return "明日までに資料を送って"
```

### 9.2 TextTransformBackend

```python
class TextTransformBackend(ABC):
    @abstractmethod
    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        ...
```

Required dummy behavior:

- Accept source text.
- Accept `TextTransformPrompt`.
- Return non-empty transformed text.
- It may use a simple deterministic sentence.
- It must not implement fixed style dropdown logic.

### 9.3 VoiceOutputBackend

```python
class VoiceOutputBackend(ABC):
    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_context: VoiceContext,
        prompt: VoiceOutputPrompt,
    ) -> str:
        ...
```

Required dummy behavior:

- Accept text, `VoiceContext`, and `VoiceOutputPrompt`.
- Create a valid short silent WAV file in `outputs/dummy.wav`.
- Use `numpy` and `soundfile`.
- Return output audio path.

The dummy implementation may not perform real voice synthesis.

## 10. Pipeline requirements

Implement `KuchikaePipeline`.

Constructor arguments:

- `audio_cache`
- `voice_context_extractor`
- `stt_backend`
- `text_transform_backend`
- `voice_output_backend`

Required method:

```python
def process(
    self,
    audio_path: str,
    text_transform_prompt: TextTransformPrompt,
    voice_output_prompt: VoiceOutputPrompt,
) -> PipelineResult:
    ...
```

Required order:

1. Add utterance to `AudioCache`.
2. Extract `VoiceContext` from reference audio.
3. Transcribe audio.
4. Transform text with `TextTransformPrompt`.
5. Synthesize audio with `VoiceContext` and `VoiceOutputPrompt`.
6. Return `PipelineResult`.

Measure:

- STT seconds
- text transform seconds
- voice output seconds
- total seconds

## 11. Gradio UI requirements

`app.py` must expose a one-screen app.

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

Do not expose model internals in the main UI.
Do not add fixed style dropdown as the primary UI.

## 12. Prompt files

Create:

`prompts/text_transform_default.txt`

```text
内容、数字、日時、固有名詞、否定条件は保ちつつ、ユーザーの指示した言い方に変換してください。
```

`prompts/voice_output_default.txt`

```text
元の話者の声質に近づけ、自然な声で話してください。
```

## 13. Required tests

### 13.1 `test_audio_cache.py`

- Add an utterance.
- Assert latest utterance path is set.
- Assert reference audio path is set.

### 13.2 `test_voice_context.py`

- Extract context from a provided reference path.
- Assert result is `VoiceContext`.
- Assert `ready is True`.

### 13.3 `test_text_transform_dummy.py`

- Run dummy transformer.
- Assert output is a non-empty string.

### 13.4 `test_voice_output_dummy.py`

- Run dummy voice output.
- Assert output WAV exists.
- Assert it can be read by `soundfile`.

### 13.5 `test_pipeline_dummy.py`

- Create temporary dummy WAV.
- Run full pipeline.
- Assert:
  - source text exists
  - transformed text exists
  - output audio path exists
  - voice_ready is true
  - latency fields are non-negative

## 14. Forbidden implementation choices

Claude must not:

- redesign the UI into a style dropdown app
- use `StyleTransferBackend` as the main name
- use `style` as the primary user control
- connect generic TTS without `VoiceContext`
- make `VoiceContext` optional in `VoiceOutputBackend`
- add real model dependencies in the first scaffold
- vendor external model repositories
- commit generated audio, model files, runs, or caches
- add database, authentication, sharing, streaming, or translation

## 15. Completion response required from Claude

After implementation, Claude must report:

1. File tree.
2. Commands to run.
3. Test status.
4. Architecture summary.
5. Any assumptions.
6. Anything intentionally not implemented.
