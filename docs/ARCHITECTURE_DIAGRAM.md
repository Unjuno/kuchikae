# Kuchikae — Architecture Diagram

## 1. Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Kuchikae Pipeline                                │
│                  Audio → STT → TextTransform → VoiceOutput                 │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐   ┌──────────────┐   ┌─────────────┐   ┌─────────────────┐   ┌────────────────┐
  │ Audio     │   │ VoiceContext │   │ STTBackend  │   │ TextTransform   │   │ VoiceOutput    │
  │ Input     │   │ Extractor    │   │             │   │ Backend         │   │ Backend        │
  │           │──>│              │──>│ (transcribe)│──>│                 │──>│                │
  │ .wav      │   │ VoiceContext │   │ source_text │   │ transformed     │   │ output_audio │
  └──────────┘   └──────────────┘   └─────────────┘   │ text            │   │ (path)         │
                                                       │                 │   └────────────────┘
                                  ┌──────────────────┐│                 │          ↓
                                  │ VoiceContext     ││                 │   ┌────────────────┐
                                  │ - ref audio path ││                 │   │ Output Audio   │
                                  │ - speaker embed  ││                 │   │ (WAV file)     │
                                  │ - prosody profile││                 │   └────────────────┘
                                  │ - ready          ││                 │
                                  └──────────────────┘│                 │
                                                       └───── Text-      │
                                                            Transform-    │
                                                             Prompt       │
                                                                 ↓        │
                                                          ┌─────────────────┐
                                                          │ VoiceOutput     │
                                                          │ Prompt (instr.) │
                                                          └─────────────────┘


  ┌──────────────────────────────────────────────────────────────────────┐
  │                          PipelineResult                               │
  │                                                                      │
  │  • source_text       (str)                                           │
  │  • transformed_text  (str)                                           │
  │  • output_audio_path   (str)                                         │
  │  • text_transform_prompt (str)                                       │
  │  • voice_output_prompt     (str)                                     │
  │  • voice_ready             (bool)                                    │
  │  • latency                 (LatencyReport)                           │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

## 2. Module Structure

```
kuchikae/
│
├── types.py              # Data definitions (shared by all modules)
│   ├── ProsodyProfile
│   ├── VoiceContext
│   ├── TextTransformPrompt
│   ├── VoiceOutputPrompt
│   ├── LatencyReport
│   └── PipelineResult
│
├── pipeline.py           # Orchestration — no model-specific logic
│   ├── KuchikaePipeline (process → PipelineResult)
│   └── create_pipeline(backend_config)  # factory: selects backends by config
│
├── audio_cache.py        # Path storage only
│   └── AudioCache
│       ├── add_utterance(audio_path) → None
│       ├── get_latest_utterance_path() → str | None
│       └── get_reference_audio_path()  → str | None
│
├── voice_context.py      # Build VoiceContext from reference audio
│   └── VoiceContextExtractor
│       └── extract(reference_audio_path) → VoiceContext
│           v0.1: checks path exists → ready=True
│
├── stt.py                # Speech-to-text — abstract interface
│   ├── STTBackend (ABC)
│   │   └── transcribe(audio_path) → str
│   ├── DummySTTBackend       → "明日までに資料を送って"
│   └── FasterWhisperSTTBackend → real STT via faster-whisper
│           env: WHISPER_MODEL_SIZE (tiny/base/small/medium/large-v3)
│
├── text_transform.py     # Text transformation — abstract interface
│   ├── TextTransformBackend (ABC)
│   │   └── transform(text, prompt) → str
│   ├── DummyTextTransformBackend     → deterministic prefix transform
│   ├── RuleTextTransformBackend      → rule-based desu-masu/plain converter
│   └── GPTTextTransformBackend       → OpenAI API (fallback: Dummy)
│           env: OPENAI_API_KEY, model="gpt-oss"
│
├── voice_output.py       # Voice output — abstract interface
│   ├── VoiceOutputBackend (ABC)
│   │   └── synthesize(text, voice_context, prompt) → str  (output path)
│   ├── DummyVoiceOutputBackend     → silent WAV file
│   └── OpenVoiceOutputBackend      → real voice cloning via OpenVoice v2
│           env: OPENVOICE_READY=1, KUCHIKAE_OPENVOICE_PATH
│
├── timing.py             # Timing utilities (Timer context manager)
│   ├── Timer
│   └── now_ms()
│
└── app.py                # Thin Gradio wrapper — no model logic
    ├── run(audio_path, text_prompt, voice_prompt) → PipelineResult
    └── gr.Blocks UI with: audio input, 2 prompt boxes, outputs


prompts/
├── text_transform_default.txt   → "内容、数字…"
└── voice_output_default.txt     → "元の話者の声質に…

outputs/
└── dummy.wav                    (generated during pipeline)

tests/
├── test_audio_cache.py          # AudioCache path operations
├── test_voice_context.py        # VoiceContextExtractor readiness logic
├── test_text_transform_dummy.py # DummyTextTransformBackend output validation
├── test_voice_output_dummy.py   # DummyVoiceOutputBackend WAV validity check
└── test_pipeline_dummy.py       # Full pipeline integration (8 tests)


External Models (outside the repo):
┌───────────────────────────────┐  ┌───────────────────────────────┐
│ /Users/taka/repos/OpenVoice   │  │ .cache/faster-whisper         │
│ ├── checkpoints/              │  │   └── small.en.pt             │
│ │   ├── base_speakers/EN/   │  └────────────────────────────────┘
│ │   │   ├── config.json      │  (lazy download on first transcribe)
│ │   │   ├── checkpoint.pth   │
│ │   │   └── en_default_se.pth│
│ │   └── converter/           │
│ │       ├── config.json      │
│ │       └── checkpoint.pth   │
└───────────────────────────────┘


Backend Selection via create_pipeline(backend_config):

  backend_config = {
      "stt_backend":        "dummy" | "faster_whisper",    # auto-detects
      "text_transform_backend": "rule" | "gpt_oss",         # default: rule
      "voice_output_backend":   "dummy" | "openvoice",       # default: dummy
  }

  • If OPENAI_API_KEY is missing → GPT backend falls back to DummyTextTransformBackend
  • If OpenVoice repo exists + OPENVOICE_READY=1 → uses OpenVoiceOutputBackend
  • All backends gracefully degrade to dummies on import errors


Key Architectural Rules:
───────────────────────

  1. Pipeline and UI contain NO model-specific logic.
     Model decisions live entirely in backend implementations.

  2. VoiceOutputBackend.synthesize() REQUIRES both VoiceContext and VoiceOutputPrompt.
     (This is the most important boundary — see SPEC §7, ACCEPTANCE §6)

  3. Text transformation ≠ voice output. Two separate layers with two prompts.

  4. External model repos live OUTSIDE the Kuchikae repo.
     The repository contains adapters, not copied models.


Dependencies:
─────────────

  v0.1 scaffold (light):
    • gradio
    • numpy
    • soundfile
    • pytest

  Real backends (heavy — installed as needed):
    • faster-whisper      → STT backend
    • torch, torchaudio   → OpenVoice voice cloning
    • httpx               → GPT text transform API


## 3. 音質の最適化戦略

### Multi-frame Tone Color Analysis
```
┌───────────────────────────────────────────────────┐
│              _extract_multi_frame_se()            │
│                                                   │
│  Reference Audio ─→ [Chunk1] [Chunk2] ... [ChunkN]  │
│                          ↓           ↓          ↓   │
│                  SE1      SE2     ...    SEN      │
│                          ↓           ↓          ↓   │
│                      Mean(SE) = ToneColorEmbedding │
└───────────────────────────────────────────────────┘

Effect: Captures voice characteristics that change over time (pitch, energy).
        Better quality than single-frame extraction.


### Lazy Model Loading (Checkpoint Caching)
```
┌───────────────────────────────────────────────────┐
│              _ensure_models_loaded()              │
│                                                   │
│  First call:                                       │
│    → Load BaseSpeakerTTS checkpoint (~50ms)       │
│    → Load ToneColorConverter checkpoint (~100ms)   │
│    → Cache in self._base_tts, self._converter     │
│                                                   │
│  Subsequent calls:                                 │
│    → Return cached models (0ms)                   │
└───────────────────────────────────────────────────┘

Effect: Eliminates repeated checkpoint loading. Startup time reduced by ~150ms per call.


### Debug Logging
```
┌───────────────────────────────────────────────────┐
│              _log() / print()                     │
│                                                   │
│  [OpenVoice] Synthesizing: text=...               │
│  [OpenVoice] Reference audio loaded                │
│  [OpenVoice] Multi-frame SE extracted (N frames)   │
│  [OpenVoice] Output saved to outputs/              │
└───────────────────────────────────────────────────┘

Effect: Convenient debugging in Gradio and CLI without modifying code.


## 4. ワンボタンモード — Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    ONE-BUTTON MODE                           │
│                                                              │
│  [Record/Upload] ─→ run_one_button(audio_path)              │
│        ↓                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ DEFAULT PROMPTS (loaded from file):                     │ │
│  │ TextTransformPrompt = "内容、数字…"                      │ │
│  │ VoiceOutputPrompt     = "元の話者の声質に…"              │ │
│  └─────────────────────────────────────────────────────────┘ │
│        ↓                                                     │
│  Pipeline.process(audio, text_prompt, voice_prompt)         │
│        ↓                                                     │
│  [Source Text]  [Transformed Text]                           │
│  [▶ Output Audio]                                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘


## 5. モデルの選択とパラメーター

### STT (Speech → Text)
| Parameter              | Value     | Effect                                    |
|------------------------|-----------|------------------------------------------|
| WHISPER_MODEL_SIZE     | "small"   | Model size: tiny(39MB)→large-v3(3GB)     |
| device                 | "cpu"     | CPU/GPU processing                        |
| compute_type           | "int8"    | Quantization — 2x speed with quality      |
| language               | "ja"      | Language (auto = slower initial detection) |

### TextTransform
| Parameter        | Value          | Effect                                  |
|------------------|---------------|----------------------------------------|
| instruction      | User input     | Prompt controlling text transformation   |
| preserve_meaning | True           | Keep numbers, dates, names intact       |
| max_output_chars | 220            | Max output length (affects API cost)    |

### VoiceOutput (OpenVoice)
| Parameter                  | Value              | Effect                              |
|----------------------------|-------------------|-------------------------------------|
| KUCHIKAE_OPENVOICE_PATH    | env var or default | Path to OpenVoice repo            |
| base_tts.speaker           | "en_default"       | Speaker preset                      |
| base_tts.language          | "English"         | TTS language                        |
| style_se (multi-frame)     | Mean of frames    | Tone color embedding from reference  |
| emotion                    | Optional           | Emotion-aware tone adjustment        |


```
