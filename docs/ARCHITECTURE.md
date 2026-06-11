# Kuchikae v0.1 Architecture

## 1. Fixed pipeline

Kuchikae v0.1 uses this fixed pipeline:

```text
Audio Input
  -> AudioCache
  -> VoiceContextExtractor
  -> STTBackend
  -> TextTransformBackend
  -> VoiceOutputBackend
  -> Output Audio
```

The pipeline must be prompt-conditioned and voice-conditioned from the first scaffold.

Do not implement Kuchikae as:

```text
Audio -> STT -> fixed style dropdown -> generic TTS
```

That is the wrong product.

## 2. Responsibility boundaries

| Component | Responsibility | Model logic allowed? |
|---|---|---:|
| `AudioCache` | Store utterance audio and reference audio paths | No |
| `VoiceContextExtractor` | Build `VoiceContext` from reference audio | Later |
| `STTBackend` | Audio to transcript | Later |
| `TextTransformBackend` | Text + prompt to transformed text | Later |
| `VoiceOutputBackend` | Text + VoiceContext + prompt to output audio | Later |
| `KuchikaePipeline` | Orchestrate components | No |
| `app.py` | UI only | No |

The pipeline and UI must not contain model-specific logic.

## 3. Required module structure

```text
kuchikae/
  app.py
  flake.nix
  pyproject.toml
  README.md
  .gitignore

  kuchikae/
    __init__.py
    types.py
    audio_cache.py
    audio_key.py
    voice_context.py
    stt.py
    text_transform.py
    voice_output.py
    pipeline.py
    processing_cache.py
    counting_backends.py
    logging.py
    ui.py
    web.py
    cli.py

  prompts/
    text_transform_default.txt
    text_transform_polite.txt
    text_transform_casual.txt
    text_transform_summarize.txt
    voice_output_default.txt

  outputs/
    .gitkeep

  tests/
    test_audio_cache.py
    test_voice_context.py
    test_pipeline_dummy.py
    test_text_transform_dummy.py
    test_voice_output_dummy.py
    test_performance.py
    test_audio.py
    test_app.py
```

## 4. Data types

`kuchikae/types.py` must define these dataclasses.

```python
from dataclasses import dataclass
from typing import Any, Optional, NamedTuple
import os


@dataclass
class VoiceContext:
    reference_audio_path: str
    ready: bool
    speaker_embedding: Optional[Any] = None
    prosody_profile: Optional[Any] = None


@dataclass
class TextTransformPrompt:
    instruction: str

    @classmethod
    def from_file(cls, path: str | None = None) -> "TextTransformPrompt":
        ...


@dataclass
class VoiceOutputPrompt:
    instruction: str

    @classmethod
    def from_file(cls, path: str | None = None) -> "VoiceOutputPrompt":
        ...


@dataclass
class PipelineResult:
    output_audio_path: str
    source_text: str = ""
    transformed_text: str = ""


class AudioCacheKey(NamedTuple):
    path: str
    size: int
    mtime: float

    @classmethod
    def from_file(cls, path: str) -> "AudioCacheKey":
        ...


@dataclass
class AudioSegment:
    start_sec: float
    end_sec: float
    samples: object
    sample_rate: int
```

## 5. AudioCache

`AudioCache` stores paths, not model features.

Required methods:

```python
class AudioCache:
    def add_utterance(self, audio_path: str) -> None:
        ...

    def get_latest_utterance_path(self) -> str | None:
        ...

    def get_reference_audio_path(self) -> str | None:
        ...
```

v0.1 may use the same audio file as both latest utterance and reference audio.

Future versions may implement a rolling reference window. v0.1 must not overbuild this.

## 5b. AudioKey and ProcessingCache

`AudioKey` provides a stable identity for an audio file (path + size + mtime).

```python
class AudioKey:
    def __init__(self, path: str, size: int, mtime: float): ...
    def __eq__(self, other) -> bool: ...
    def __hash__(self) -> int: ...

class AudioKeyFromCacheKey:
    def __call__(self, cache_key: AudioCacheKey) -> AudioKey: ...
```

`ProcessingCache` caches STT results and VoiceContext per `AudioKey`:

```python
class ProcessingCache:
    def get_stt(self, audio_key: AudioKey) -> str | None: ...
    def set_stt(self, audio_key: AudioKey, text: str) -> None: ...
    def get_voice_context(self, audio_key: AudioKey) -> VoiceContext | None: ...
    def set_voice_context(self, audio_key: AudioKey, context: VoiceContext) -> None: ...
```

Pipeline uses `ProcessingCache` to avoid redundant STT and VoiceContext extraction when the same audio is processed with different prompts.

## 6. VoiceContextExtractor

Required interface:

```python
class VoiceContextExtractor:
    def extract(self, reference_audio_path: str | None) -> VoiceContext:
        ...
```

v0.1 dummy behavior:

- If a reference path exists or is provided, return `VoiceContext(..., ready=True)`.
- Do not compute real speaker embeddings.
- Do not require a model.

## 7. STTBackend

Required interface:

```python
from abc import ABC, abstractmethod


class STTBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        ...
```

v0.1 implementation:

```python
class DummySTTBackend(STTBackend):
    def transcribe(self, audio_path: str) -> str:
        return "明日までに資料を送って"
```

Future candidate:

- `FasterWhisperSTTBackend`

Do not import `faster-whisper` in the initial scaffold.

## 8. TextTransformBackend

Required interface:

```python
from abc import ABC, abstractmethod
from kuchikae.types import TextTransformPrompt


class TextTransformBackend(ABC):
    @abstractmethod
    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        ...
```

v0.1 dummy behavior may return a deterministic transformed sentence. It must accept the prompt object, even if the dummy implementation does not use all fields.

Important:

- Do not implement fixed style dropdown logic.
- Do not hard-code `polite`, `casual`, `rpg` as the primary control.
- Prompt presets may be added later, but the backend interface is prompt-based.

### Available implementations

| Backend | Description |
|---------|-------------|
| `DummyTextTransformBackend` | Returns placeholder `[transformed] {text}` |
| `RuleTextTransformBackend` | Rule-based verb ending conversion (desu-masu/plain) |
| `PromptedRuleTextTransformBackend` | Rule-based with prompt type detection; loads few-shot templates from `prompts/` (default) |
| `OllamaTextTransformBackend` | Calls local Ollama API (requires Ollama server) |
| `GPTTextTransformBackend` | Calls OpenAI-compatible API (requires API key) |

Prompt templates with few-shot examples are stored in `kuchikae/prompts/`:
- `text_transform_polite.txt` — 6 examples for polite conversion
- `text_transform_casual.txt` — 6 examples for casual conversion
- `text_transform_summarize.txt` — 5 examples for summarization
- `text_transform_default.txt` — template with `{instruction}` and `{text}` placeholders

## 9. VoiceOutputBackend

Required interface:

```python
from abc import ABC, abstractmethod
from kuchikae.types import VoiceContext, VoiceOutputPrompt


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

This is the most important architectural boundary.

Rules:

1. `voice_context` is required.
2. `prompt` is required.
3. The method returns an output audio file path.
4. Dummy implementation must create a valid WAV file.
5. Real implementations must remain behind this backend.

Future candidates:

- `OpenVoiceBackend`
- `GPTSoVITSBackend`
- `CosyVoiceBackend`
- `F5TTSBackend`
- `TTSPlusVCBackend`

Do not import heavy model packages in the initial scaffold.

## 10. Pipeline

Required order:

```text
1. Add utterance to AudioCache
2. Extract VoiceContext from reference audio
3. Transcribe audio with STTBackend
4. Transform transcript with TextTransformBackend and TextTransformPrompt
5. Synthesize with VoiceOutputBackend, VoiceContext, and VoiceOutputPrompt
6. Return PipelineResult
```

Sketch:

```python
class KuchikaePipeline:
    def __init__(
        self,
        stt_backend: STTBackend | None = None,
        text_transform_backend: TextTransformBackend | None = None,
        voice_output_backend: VoiceOutputBackend | None = None,
        processing_cache: ProcessingCache | None = None,
    ) -> None:
        ...

    def process(
        self,
        audio_path: str,
        text_transform_prompt: TextTransformPrompt,
    ) -> PipelineResult:
        ...

    def process_stream(
        self,
        audio_path: str,
        text_transform_prompt: TextTransformPrompt,
    ) -> Generator[tuple[str, str | None, str | None, str | None], None, None]:
        """Progressive output for UI streaming.
        Yields: (status, source_text, transformed_text, output_audio_path)
        Status: "STT" | "TXT" | "VOX" | "DONE"
        """
        ...
```

## 11. UI architecture

`app.py` must be a thin Gradio wrapper.

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

Do not expose backend model settings in the UI. Backend configuration belongs in config files or later CLI/dev settings.

## 12. External models policy

External model repositories must not be vendored into Kuchikae v0.1.

Allowed later pattern:

```yaml
external_models:
  openvoice_path: "../OpenVoice"
  gpt_sovits_path: "../GPT-SoVITS"
  cosyvoice_path: "../CosyVoice"
```

The Kuchikae repository should contain adapters, not copied model repositories or model weights.

### Vendor directory (optional)

For fully self-contained distribution, optional `_vendor/` directory may contain minimal copies of runtime-required model code (e.g., `irodori_tts/inference_runtime.py`). This is not required for v0.1 development but enables single-repo distribution.

```
kuchikae/_vendor/
  irodori_tts/
    inference_runtime.py
    config.py
    ...
```

## 13. v0.1 dependency policy

The initial scaffold must not include:

- torch
- transformers
- faster-whisper
- OpenVoice
- GPT-SoVITS
- CosyVoice
- F5-TTS
- RVC
- any large model dependency

Only dummy backends are required for the first implementation.
