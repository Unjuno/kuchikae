# Model Survey

Environment:
- Host: macOS 26.5.1 on arm64
- Python: 3.11.15
- torch: 2.12.0
- MPS: available
- CUDA: unavailable
- ffmpeg: available
- uv: available

This survey focuses on the models that are directly relevant to Kuchikae's current pipeline:
- Japanese ASR for the source transcript
- Japanese TTS / voice cloning for the output voice
- codec / latent audio representation for TTS runtime

## Shortlist

| Area | Model | License | Size | Japanese | Mac CPU | MPS | CUDA | CTranslate2/MLX/ONNX | Decision | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| ASR | `japanese-asr/distil-whisper-large-v3-ja-reazonspeech-large` | unclear in card; alias of `kotoba-tech/kotoba-whisper-v1.0` | 0.8B | yes | yes via Transformers | possible via torch backend, not validated | yes via Transformers | no validated CT2/MLX/ONNX path | keep | Strong Japanese Whisper-family candidate; good for quality experiments, but slower/heavier and license metadata needs care. |
| ASR | `TKU410410103/hubert-large-japanese-asr` | Apache-2.0 | 0.3B | hiragana-only output | yes via Transformers | possible, not validated | yes | no validated CT2/MLX/ONNX path | try | Smaller than Whisper, but output is hiragana-only so Kuchikae needs normalization/post-processing. |
| ASR | `kotoba-tech/kotoba-whisper-v1.0-faster` | not rechecked in this run | 0.756B | yes | yes via faster-whisper | yes via faster-whisper | yes via faster-whisper | faster-whisper / whisper.cpp weights advertised | try | The repo advertises faster-whisper weights and an inference-speed benchmark on Mac; worth testing because it can slot into the existing backend. |
| ASR | `kotoba-tech/kotoba-whisper-v2.1` | Apache-2.0 | 0.756B | yes | not practical on this Mac | not validated | yes via Transformers | no validated CT2/MLX/ONNX path | reject for this Mac | The repo is a Whisper-family Transformers model with punctuation support, but the current Transformers CTC backend cannot load it, and the dedicated Whisper backend smoke did not complete in a practical window on this Mac CPU. |
| ASR | `quantumcookie/anime-whisper-ct2` | not rechecked in this run | CT2 Whisper repo | yes | yes via CT2 | not validated | yes via CT2 | CTranslate2 weights present; model.bin listed | reject for this Mac | The repo exposes CT2 weights, but local smoke runs did not complete in a practical window on this Mac. |
| ASR | `quantumcookie/anime-whisper-ct2-fp16` | not rechecked in this run | CT2 Whisper repo | yes | yes via CT2 | not validated | yes via CT2 | CTranslate2 weights present; model.bin listed | reject for this Mac | The fp16 CT2 variant also failed to complete in a practical benchmark window on this Mac. |
| ASR | `quantumcookie/anime-whisper-ct2-int8` | not rechecked in this run | CT2 Whisper repo | yes | yes via CT2 | not validated | yes via CT2 | CTranslate2 weights present; model.bin listed | reject for this Mac | The int8 CT2 variant also failed to complete in a practical benchmark window on this Mac. |
| ASR | `reazon-research/reazonspeech-nemo-v2` | Apache-2.0 | 619M | yes | yes via NeMo | not validated | yes | no validated CT2/MLX/ONNX path | reject | `uv sync --extra nemo-stt` failed on this Mac because `onnx==1.12.0` could not build on macOS arm64. The model remains interesting, but the local dependency stack is too costly. |
| ASR | `prj-beatrice/japanese-hubert-base-phoneme-ctc-v3` | not fully verified in this run | 94.4M | phoneme/CTC oriented | yes via Transformers | possible, not validated | yes | no validated CT2/MLX/ONNX path | reject for now | Promising for research, but output form is too far from Kuchikae's current text pipeline without extra normalization work. |
| TTS | `Aratako/Irodori-TTS-500M-v3` | MIT | 0.5B | yes | yes | not validated | yes | runtime is Python-based; no CT2/MLX/ONNX path validated | adopt | Current backend works, has good Japanese support, and warm latency is materially better at `num_steps=6` than `10` on this Mac while preserving non-empty audio output. |
| Codec | `Aratako/Semantic-DACVAE-Japanese-32dim` | MIT | codec model | yes | yes | not validated | yes | no validated CT2/MLX/ONNX path | adopt | Current codec for Irodori. The 32-dim latent is explicitly intended for compact, fast Japanese reconstruction. |

## Notes from the official model pages

- `japanese-asr/distil-whisper-large-v3-ja-reazonspeech-large`
  - Hugging Face model page shows it as an ASR Whisper-family model with Transformers and Safetensors.
  - The page states it is an alias of `kotoba-tech/kotoba-whisper-v1.0`.
  - Model size is listed as 0.8B params.
  - Inference Providers are not available on the page.
  - License metadata was not clearly surfaced in the page snippet used during this run, so treat licensing as unresolved until the underlying model card is checked in more detail.

- `TKU410410103/hubert-large-japanese-asr`
  - The model card states it is a fine-tuned version of `rinna/japanese-hubert-large`.
  - It was initially fine-tuned on ReazonSpeech small, then on Common Voice 11.0.
  - The card explicitly says it can only predict Hiragana.
  - Apache-2.0 license is listed.
  - In this run, the transformers backend was able to resolve the repo cache, but the benchmark job did not complete in a practical time window for this machine, so it remains a "try" candidate only.

- `kotoba-tech/kotoba-whisper-v1.0-faster`
  - The model page advertises faster-whisper compatible weights and a Mac benchmark table.
  - This makes it the most direct "drop-in" candidate for the current backend, but it still needs a successful local benchmark pass before it can be adopted.
  - This run did not complete a full benchmark yet, so it remains a "try" candidate.

- `kotoba-tech/kotoba-whisper-v2.1`
  - The model page is a Whisper-family Transformers model with punctuation postprocessing.
  - It is not compatible with the current CTC-only Transformers backend.
  - After adding a dedicated Whisper backend, the short smoke run still did not complete in a practical window on this Mac CPU, so it remains rejected for this environment.

- `quantumcookie/anime-whisper-ct2`, `quantumcookie/anime-whisper-ct2-fp16`, `quantumcookie/anime-whisper-ct2-int8`
  - The hub exposes CT2-compatible weights and the local environment has both `ctranslate2` and `faster_whisper` installed.
  - In practice, all three variants failed to complete a smoke benchmark in a practical window on this Mac.
  - No JSON benchmark artifact was produced, so they are not currently usable as drop-in replacements here.

- `Aratako/Irodori-TTS-500M-v3`
  - The model page describes Japanese TTS with emoji/style-driven control.
  - It uses `Aratako/Semantic-DACVAE-Japanese-32dim` as the codec.
  - Model size is 0.5B params.
  - MIT license is listed.
  - On this Mac, warm latency improved materially at `num_steps=6` versus `10` while keeping the output duration unchanged at `3.72s`.

- `Aratako/Semantic-DACVAE-Japanese-32dim`
  - The model page says the latent dimension was reduced from 128 to 32 while keeping Japanese speech reconstruction quality competitive.
  - MIT license is listed.
  - The page includes evaluation notes showing competitive UTMOSv2 scores on Japanese subsets.

- `reazon-research/reazonspeech-nemo-v2`
  - The page says it is trained on ReazonSpeech v2.0.
  - It supports long-form Japanese audio clips up to several hours.
  - The page lists Apache-2.0 and a 619M parameter count.
  - This makes it a candidate for long recordings, but the heavier NeMo dependency stack reduced practical value for the current app on this Mac.
  - Local environment note: `uv sync --extra nemo-stt` failed because `onnx==1.12.0` could not build on macOS arm64.

## Current conclusion

For this repository as it exists now:
- ASR default should be `tiny` for speed on Mac CPU, with `small` and heavier Whisper-family models left as opt-in experiments. NeMo-based models are currently not practical on this Mac due to the `onnx` build failure in the optional dependency stack.
- CT2 Whisper-family variants from `quantumcookie/anime-whisper-ct2*` are not practical on this Mac as current benchmark attempts do not complete in a bounded window.
- TTS should stay on `Irodori-TTS-500M-v3` with `num_steps=6` as the default on this Mac unless a larger benchmark run proves otherwise.
- The codec should stay on `Semantic-DACVAE-Japanese-32dim`.
