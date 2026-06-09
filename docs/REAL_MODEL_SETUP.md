# Real Model Setup Guide

## Quick start (OpenVoice — Phase 3)

```bash
cd /Users/taka/repos/OpenVoice
git lfs install
git lfs pull  # download checkpoint files (~600 MB).
OPENVOICE_READY=1 uv run python scripts/smoke_e2e.py
```

If `git lfs` is not available, the checkpoints will be downloaded lazily on first use.

## Model candidates evaluated

| Model | Voice cloning | Japanese | Mac ARM64 | Setup | License | Decision |
|---|---|---|---|---|---|---|
| **OpenVoice v2** | ✅ ref audio | ✅ good | ✅ cpu | git clone | MIT | **Phase 3** |
| F5-TTS | ✅ speaker | ✅ good | ✅ mps | pip install | Apache-2.0 | next candidate |
| E2-TTS | ✅ speaker | ✅ good | ✅ | pip install | Apache-2.0 | viable |
| GPT-SoVITS | ✅ great | ✅ excellent | ⚠️ heavy | pip install + model | MIT | great quality, heavier |
| CosyVoice | ✅ speaker | ✅ excellent | ⚠️ heavy | pip install | Apache-2.0 | next candidate |
| XTTS-v2 | ✅ speaker | ✅ excellent | ✅ cpu | pip install | MPL-2.0 | good fallback |

## faster-whisper (Phase 1 — STT)

```bash
uv pip install faster-whisper
WHISPER_MODEL_SIZE=small uv run python scripts/smoke_real_stt.py
```

Models: tiny → base → small (~150MB) → medium (~800MB) → large-v3 (~3GB).
Japanese quality is good with `medium` or `large-v3`.

## GPT-oss TextTransform (Phase 2 — API-backed LLM)

```bash
export OPENAI_API_KEY=sk-...
uv run python -c "from kuchikae.text_transform import GPTTextTransformBackend; print(GPTTextTransformBackend().transform('こんにちは', None))"
```

## No-API fallbacks

All backends gracefully fall back to `Dummy*` when dependencies are missing:
- `FasterWhisperSTTBackend` → raises clear error. Use `create_pipeline()` for auto-detection.
- `GPTTextTransformBackend` → falls back to `DummyTextTransformBackend`.
- `OpenVoiceOutputBackend` → requires OpenVoice repo + checkpoints.
- `RuleTextTransformBackend` works with **no external models**.

## External model location policy

External model repos live **outside** the Kuchikae repository:
- `/Users/taka/repos/OpenVoice` (cloned via git)
- Model weights in `.cache/` or inside the cloned repo's `checkpoints/`.
