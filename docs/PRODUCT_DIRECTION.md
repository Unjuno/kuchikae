# PRODUCT DIRECTION — Beyond Dummy Scaffold → Real Model Prototype

## 1. Direction (confirmed)

Kuchikae must move beyond its current dummy scaffold toward a real-model prototype, while preserving the **prompt-conditioned** and **voice-conditioned** concept that defines the product identity.

The scaffolding is not the product; it is a temporary bridge until real models take their place in each phase of the pipeline (STT → TextTransform → VoiceOutput).

## 2. Dummy backend = fallback, not final goal

Dummy backends exist to keep Kuchikae runnable and testable without any external dependencies:
- `DummySTTBackend` returns a fixed string — correct as scaffolding, incorrect as product output.
- `DummyTextTransformBackend` returns the input text unchanged. Correct for development, wrong for real use.
- `DummyVoiceOutputBackend` writes a silent WAV file. This is the most critical gap: VoiceOutput is where users perceive quality directly.

When all backends are dummy, Kuchikae runs but produces no meaningful voice output. The product direction requires replacing these dummies with real models while keeping the same interface — the pipeline and UI remain unchanged.

## 3. Test tiers

**Baseline tests (run without models):**
Tests in `tests/` must pass on any machine with Python, regardless of whether Whisper, OpenVoice, or other model dependencies are installed. These validate backend interfaces, pipeline orchestration, AudioCache logic, and voice context extraction.

**Smoke tests (require real models):**
Scripts under `scripts/smoke_*.py` exercise each phase end-to-end:
- `smoke_real_stt.py` — transcribes test audio through FasterWhisper
- `smoke_e2e.py` — full pipeline with OpenVoice voice cloning
- These are manual gates, not CI blockers.

## 4. Primary quality target: VoiceOutput

Of the three phases (STT → TextTransform → VoiceOutput), **VoiceOutput is the primary quality signal**:
- Users perceive STT as "transcription" and TextTransform as "rewriting." Both have many mature alternatives.
- VoiceOutput — where Kuchikae's dual-conditioning (prompt + voice reference) shines — is its differentiator.

OpenVoice v2 is currently selected as the primary VoiceOutput backend, with F5-TTS and CosyVoice as candidates for the future. The quality goal is: given a user's uploaded audio and free-form instructions, produce voice output that sounds like the original speaker delivering the transformed text.

## 5. Model integration strategy

Models enter Kuchikae exclusively through **backend adapters**:
- `STTBackend` → `FasterWhisperSTTBackend` (whisper model weights external)
- `TextTransformBackend` → `GPTTextTransformBackend` (LLM API, no local weight dependency beyond the rule-based fallback)
- `VoiceOutputBackend` → `OpenVoiceOutputBackend` (checkpoint files external)

The pipeline (`pipeline.py`) and UI (`app.py`) contain zero model-specific logic. Model decisions live entirely in backend implementations. When a new model is selected, only the adapter changes — not the orchestration or interface layers.

## 6. External models = outside the repository

No model weights, checkpoints, or external repositories are committed to Kuchikae:
- OpenVoice lives at `$KUCHIKAE_OPENVOICE_PATH` (cloned via git, LFS pull).
- Whisper models live in `~/.cache/faster-whisper/`.
- Model weights downloaded lazily on first use.

This keeps the repository lean and model selections independent of Kuchikae's own versioning.

## 7. UI remains free-form prompt based

Kuchikae's product identity is two-layer prompting:
1. **Text transform prompt** — "what to say" (content, numbers, structure)
2. **Voice output prompt** - "how to sound" (tone, emotion, speaker characteristics)

The one-button mode simplifies the UI but preserves these same two concepts through default presets loaded from `prompts/`. The core experience remains free-form: users upload audio and give natural-language instructions — no dropdowns or fixed templates required.
