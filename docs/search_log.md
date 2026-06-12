# Search Log

## Hugging Face / Model Search

### Queries
- `japanese asr`
- `japanese whisper`
- `distil whisper japanese`
- `kotoba whisper`
- `reazonspeech whisper`
- `japanese tts`
- `japanese voice cloning`
- `japanese reference audio tts`
- `irodori tts`
- `semantic dacvae japanese`
- `mlx whisper`
- `faster whisper`
- `ctranslate2 japanese`

### Official pages inspected
- `japanese-asr/distil-whisper-large-v3-ja-reazonspeech-large`
- `kotoba-tech/kotoba-whisper-v1.0`
- `kotoba-tech/kotoba-whisper-v1.0-faster`
- `kotoba-tech/kotoba-whisper-v1.0-ggml`
- `kotoba-tech/kotoba-whisper-v2.0-faster`
- `kotoba-tech/kotoba-whisper-v2.0-ggml`
- `kotoba-tech/kotoba-whisper-v2.1`
- `TKU410410103/hubert-large-japanese-asr`
- `Aratako/Irodori-TTS-500M-v3`
- `Aratako/Semantic-DACVAE-Japanese-32dim`

### Findings
- `kotoba-tech/kotoba-whisper-v2.1` is a Whisper-family Transformers model with punctuation post-processing support, but it is not a drop-in fit for the old CTC-only backend.
- `kotoba-tech/kotoba-whisper-v1.0-faster` and CT2 variants were inspected as runtime candidates; on this Mac they were not practical within the benchmark budget.
- `Aratako/Irodori-TTS-500M-v3` remains the practical TTS default.
- `Aratako/Semantic-DACVAE-Japanese-32dim` is the codec dependency used by Irodori.

## Web / Runtime Search

### Queries
- `Hugging Face Japanese ASR Whisper license size Mac CPU GPU`
- `Hugging Face Japanese TTS reference audio license size Mac CPU GPU`
- `Irodori TTS 500M v3 license model size`
- `Semantic DACVAE Japanese 32dim license`
- `faster-whisper benchmark Mac Apple Silicon int8 Whisper`
- `CTranslate2 Apple Silicon CPU benchmark`
- `MLX Whisper Japanese benchmark Mac`

### Notes
- local Ollama is available on this machine
- `hf.co/LiquidAI/LFM2.5-1.2B-JP-GGUF:Q4_K_M` is usable for text transform experiments
- LLM transform with Ollama followed prompts more usefully than the rule backends in this repo

