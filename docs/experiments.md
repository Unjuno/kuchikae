# Experiments

## Experiment 1: FasterWhisper `tiny` vs `small` on real Japanese audio

H: On this Mac, `tiny` will reduce ASR latency compared with `small` while still producing a non-empty Japanese transcript for the same input audio.

T: Run the STT benchmark on the same real Japanese audio produced by Irodori TTS, once with `WHISPER_MODEL_SIZE=tiny` and once with `WHISPER_MODEL_SIZE=small`, using the same backend settings and the same input file.

D:
- PASS if `tiny` has a lower median warm total time and returns a non-empty transcript.
- FAIL if it returns empty text or if latency is not meaningfully better.
- UNCERTAIN if latency improves but transcription quality looks unstable.

C:
- Small sample size.
- Input audio is TTS-generated, not human speech.
- Results may overfit to this one utterance.

U:
- Real-world conversational audio may behave differently.
- The benchmark measures a single short utterance, so accuracy generalization is limited.

Result:
- `tiny`: median warm total `0.2555s`, median warm RTF `0.0687`
- `small`: median warm total `0.7272s`, median warm RTF `0.1955`
- Transcript for both: `明日までに資料を送ってください`
- Decision: adopt `tiny` as the default ASR size for speed on this machine.

## Experiment 2: Irodori TTS `num_steps=4` vs `num_steps=10`

H: Reducing `num_steps` from 10 to 4 will reduce TTS latency without collapsing output quality.

T: Run the same text and the same reference audio through Irodori TTS with `num_steps=4` and `num_steps=10`, on the same Mac, with warm runs.

D:
- PASS if `num_steps=4` is at least 10% faster and still emits non-empty audio with usable duration.
- FAIL if the output is empty or clearly degraded.
- UNCERTAIN if the audio is produced but subjective quality is questionable.

C:
- The benchmark is short and sample-limited.
- The model may warm up differently depending on prior imports and runtime cache.

U:
- This is one prompt and one reference voice only.
- Cold-start load time is much larger than warm latency and can skew interpretation.

Result:
- `num_steps=4`: median warm total `1.9295s`, median warm RTF `0.5187`
- `num_steps=10`: median warm total `1.5344s`, median warm RTF `0.4125`
- Decision: keep `num_steps=10` for now; on this machine and sample it was faster and produced valid audio.

## Experiment 3: Full pipeline on `tiny` vs `small`

H: Switching the ASR default from `small` to `tiny` will reduce end-to-end pipeline latency while preserving the source transcript.

T: Run the same end-to-end pipeline on the same Japanese audio with the same text transform and same voice backend, once with `WHISPER_MODEL_SIZE=tiny` and once with `WHISPER_MODEL_SIZE=small`.

D:
- PASS if the full pipeline latency improves and the transcript stays non-empty.
- FAIL if output breaks.
- UNCERTAIN if the TTS latency dominates too much to isolate ASR improvement.

C:
- Pipeline time includes TTS, which is much larger than ASR.
- The comparison is still useful because the ASR change is visible in the total latency.

U:
- TTS latency dominates, so the end-to-end improvement is smaller than the ASR-only delta.

Result:
- `tiny` raw pipeline run: total `2.7151s`, STT `0.2571s`, voice `2.4576s`
- `small` raw pipeline run: total roughly `3.72s` in the earlier benchmark, with STT around `0.93s`
- Decision: adopt `tiny` as the default ASR size; the pipeline is materially faster.

## Additional observations

- `Irodori-TTS-500M-v3` cold load time is large on this machine, but warm synthesis is usable.
- The generated voice output is non-empty and the output files are valid WAVs.
- The TTS backend benefits from unique output file naming, which avoids overwrite races during repeated benchmarks.

## Experiment 4: Transformers Hubert ASR on Japanese TTS audio

H: `TKU410410103/hubert-large-japanese-asr` will produce a non-empty transcript on the same Japanese TTS fixture and may improve accuracy versus `tiny`, even if it is slower.

T: Run the new transformers-based backend on the same real Japanese WAV used in the earlier benchmarks, with the same warmup/run structure.

D:
- PASS if it completes and produces a non-empty transcript with acceptable latency.
- FAIL if the backend cannot complete in a practical time window or returns empty text.
- UNCERTAIN if it completes but the speed tradeoff is too large to justify on this Mac.

C:
- The model is hiragana-only, so direct comparison with the current kanji/kana output is imperfect.
- The benchmark run in this repository did not complete in a practical time window on this machine.

U:
- A longer run may eventually complete, but the job consumed wall-clock time with negligible CPU usage and did not return a result promptly.

Result:
- Attempted benchmark job was started, but it did not complete in a practical time window and was aborted.
- Decision: keep as a "try" candidate, not adopted.

## Experiment 5: Kotoba Whisper bilingual faster-whisper path

H: `kotoba-tech/kotoba-whisper-bilingual-v1.0` will expose a faster-whisper-compatible weight layout that can be benchmarked as a drop-in ASR backend.

T: Run the existing faster-whisper backend with `--model-size kotoba-tech/kotoba-whisper-bilingual-v1.0` on the same Japanese TTS fixture.

D:
- PASS if the backend loads and returns a non-empty transcript with reasonable latency.
- FAIL if the repo does not provide the expected faster-whisper weight layout or the backend cannot complete.

C:
- The repo may not contain a `model.bin` layout compatible with the current backend.
- The benchmark job failed with `Unable to open file 'model.bin'`.

U:
- The model page advertises faster-whisper weights, but the local load path used by the current backend is stricter than the page snippet suggested.

Result:
- Benchmark JSON: `benchmarks/results/stt_kotoba_bilingual.json`
- Failure: `RuntimeError: Unable to open file 'model.bin' in model '/Users/taka/.cache/huggingface/hub/models--kotoba-tech--kotoba-whisper-bilingual-v1.0/snapshots/5b2c01ec5bc234cd784e9187e74bc86582ed5944'`
- Decision: reject for the current backend path; keep as a model-card candidate only until the correct weight layout is verified.

## Experiment 6: Kotoba Whisper v1.0 faster-whisper repo

H: `kotoba-tech/kotoba-whisper-v1.0-faster` will be a drop-in faster-whisper candidate that can outperform `tiny` on quality while remaining benchmarkable on this Mac.

T: Run the existing faster-whisper backend with `--model-size kotoba-tech/kotoba-whisper-v1.0-faster` on the same Japanese TTS fixture.

D:
- PASS if the backend loads and returns a non-empty transcript with reasonable latency.
- FAIL if the repo cannot be used by the current backend path or the benchmark cannot complete.

C:
- The benchmark job was started but did not complete in a practical time window and was aborted.

U:
- The model page advertises faster-whisper weights, but the local load path used by the current backend did not complete promptly enough to benchmark on this machine.

Result:
- Attempted benchmark job was started and then aborted before completion.
- Decision: keep as a try candidate only; not adopted.

## Experiment 7: ReazonSpeech NeMo v2 as a dedicated backend

H: `reazon-research/reazonspeech-nemo-v2` will be usable as a dedicated Japanese ASR backend for long-form audio, but may be too heavy to make the default path practical on this Mac.

T: Add a backend wrapper around `nemo.collections.asr.ASRModel.from_pretrained("reazon-research/reazonspeech-nemo-v2")`, then benchmark it on the same Japanese fixture as the FasterWhisper runs.

D:
- PASS if the backend loads and produces a non-empty transcript with a practical warm latency.
- FAIL if the NeMo dependency stack is unavailable or the benchmark cannot complete.
- UNCERTAIN if transcription works but the load/runtime cost is too high for the current app.

C:
- This machine does not currently have the NeMo dependency stack installed.
- The model is 619M parameters and intended for long-form audio, so startup cost may be substantial.
- On this machine, `uv sync --extra nemo-stt` failed before benchmarking because `onnx==1.12.0` could not build on macOS arm64.

U:
- The model is promising for long recordings, but the app currently favors low-latency short-form interaction on Mac CPU.

Result:
- Dedicated backend added as `ReazonSpeechNemoASRBackend`.
- Benchmark wiring added to `benchmarks/benchmark_stt.py`.
- The optional dependency stack failed to install locally because `onnx==1.12.0` could not build.
- Decision: reject for the current Mac environment; keep as a theoretical candidate only.

## Experiment 8: Irodori TTS `num_steps=6` vs `10` on the same reference voice

H: On this Mac, `num_steps=6` will reduce warm Irodori TTS latency versus `num_steps=10` while preserving non-empty output and similar output duration.

T: Run the same Japanese text and the same fixed reference audio through Irodori TTS with `num_steps=6` and `num_steps=10`, using the same backend settings and warm runs, then compare median warm total time and output duration.

D:
- PASS if `num_steps=6` is faster and the output duration stays at `3.72s` with non-silent audio.
- FAIL if `num_steps=6` returns empty output or collapses into silence.
- UNCERTAIN if the speedup comes with obvious quality degradation that is not captured by the simple proxy metrics.

C:
- The quality check uses simple proxy metrics (`duration`, `RMS`, `peak`, `non_silence_ratio`) rather than human listening.
- This is still a single reference voice and a single sentence.

U:
- Subjective naturalness and speaker similarity are not fully measured.

Result:
- `num_steps=10`, `linear`: median warm total `1.5337s`, median warm RTF `0.4123`, output duration `3.72s`
- `num_steps=6`, `linear`: median warm total `0.7088s`, median warm RTF `0.1905`, output duration `3.72s`
- `num_steps=6`, `sway`: median warm total `1.3644s`, median warm RTF `0.3668`, output duration `3.72s`
- Proxy audio stats for recent `num_steps=6` outputs: RMS roughly `0.099` to `0.133`, peak roughly `0.639` to `0.997`, non-silence ratio roughly `0.41` to `0.51`
- Decision: adopt `num_steps=6` with `t_schedule_mode=linear` as the default on this Mac; keep `sway` as a slower alternative.

## Experiment 9: Full pipeline with TTS default switched to `num_steps=6`

H: Changing the Irodori default from `num_steps=10` to `num_steps=6` will materially reduce end-to-end pipeline latency on this Mac while keeping the output non-empty.

T: Run the same pipeline benchmark on the same synthetic Japanese fixture after changing the Irodori backend default to `num_steps=6`, then compare to the prior `pipeline_tiny_raw.json` result.

D:
- PASS if median warm total time drops while the output remains non-empty.
- FAIL if the pipeline output breaks or becomes empty.
- UNCERTAIN if the speedup is small or comes from unrelated changes.

C:
- The STT in this benchmark still has some source-text uncertainty because the generated fixture is not human speech.

U:
- The benchmark measures a fixed generated audio sample rather than a broader distribution of user speech.

Result:
- Previous pipeline baseline (`pipeline_tiny_raw.json`): median warm total `2.7151s`, median warm RTF `0.7299`
- New pipeline after TTS default change (`pipeline_tiny_default6.json`): median warm total `0.7764s`, median warm RTF `0.1553`
- Improvement: about `71%` lower median warm total time on this Mac
- Decision: keep `num_steps=6` as the default TTS setting for this machine

## Experiment 10: Irodori TTS `cfg_scale_text` / `cfg_scale_speaker` sweep at `num_steps=6 linear`

H: On this Mac, changing `cfg_scale_text` and `cfg_scale_speaker` around the default Irodori settings will materially change warm TTS latency or output duration, so a narrow sweep may uncover a better default than the current `2.0/3.0` pair.

T: Run the same Japanese text and the same reference audio through Irodori TTS with `num_steps=6`, `t_schedule_mode=linear`, and three `cfg` pairs: `1.5/2.5`, `2.0/3.0`, and `2.5/3.5`, then compare median warm total time and output duration.

D:
- PASS if one pair is meaningfully faster while still emitting non-empty audio with stable duration.
- FAIL if output becomes empty or latency regresses sharply.
- UNCERTAIN if the latency differences are negligible and only subjective quality would distinguish them.

C:
- The benchmark uses proxy metrics and one reference voice only.
- Any audible quality changes are not captured here.

U:
- The effect of these `cfg` knobs may be primarily qualitative rather than latency-related.

Result:
- `1.5/2.5`: median warm total `0.7100s`, median warm RTF `0.1908`, output duration `3.72s`
- `2.0/3.0`: median warm total `0.7096s`, median warm RTF `0.1907`, output duration `3.72s`
- `2.5/3.5`: median warm total `0.7077s`, median warm RTF `0.1902`, output duration `3.72s`
- Decision: no meaningful latency difference; keep the default `2.0/3.0` pair for now and treat these knobs as quality-only candidates.

## Experiment 11: Irodori TTS `speaker_kv_*` sweep at `num_steps=6 linear`

H: On this Mac, enabling a smaller `speaker_kv_scale` and limiting `speaker_kv_max_layers` will reduce TTS latency or at least shift the warm runtime in a measurable way compared with leaving those knobs unset.

T: Run the same Japanese text and reference audio through Irodori TTS with `num_steps=6`, `t_schedule_mode=linear`, and `speaker_kv_scale=0.5`, `speaker_kv_min_t=0.5`, `speaker_kv_max_layers=8`, then compare to the existing `num_steps=6 linear` baseline.

D:
- PASS if the warm median improves materially while output stays non-empty.
- FAIL if output breaks.
- UNCERTAIN if the runtime stays effectively unchanged.

C:
- Proxy metrics do not capture speaker similarity changes.
- The sweep only tests one speaker reference and one sentence.

U:
- These parameters may influence style fidelity more than speed.

## Experiment 12: Kotoba Whisper v2.1 on the existing CTC backend, then on a dedicated Whisper backend

H: `kotoba-tech/kotoba-whisper-v2.1` can serve as a practical Japanese ASR upgrade if the repository adds a Whisper-capable Transformers backend and the model finishes within a usable warm latency on this Mac.

T: First run the existing CTC-only Transformers backend against `kotoba-tech/kotoba-whisper-v2.1` to confirm whether the model class matches. Then run the new Whisper-specific backend against the same 5-second Japanese WAV fixture with `runs=1` and `warmup=0` to check whether the model completes promptly on CPU.

D:
- PASS if the model loads in the appropriate backend and returns non-empty text within a practical benchmark window.
- FAIL if the backend/model pairing is invalid or the run does not complete in a practical time window on this Mac.
- UNCERTAIN if transcription works but the CPU runtime is too slow for the app's latency budget.

C:
- The first attempt uses a CTC-only loader and is expected to fail for Whisper configs.
- The second attempt fixes the loader mismatch, but CPU runtime can still be too slow to justify adoption.

U:
- Model compatibility and runtime practicality are separate failure modes, so both need to be recorded.

Result:
- Existing CTC-only backend failed immediately with `WhisperConfig` vs `AutoModelForCTC` mismatch.
- Dedicated Whisper backend was added as `TransformersWhisperJapaneseASRBackend` and wired into the pipeline and STT benchmark harness.
- A short CPU smoke run on this Mac did not complete in a practical time window and was aborted.
- Decision: reject `kotoba-tech/kotoba-whisper-v2.1` for the current Mac CPU path; keep only as a future candidate if a faster runtime path becomes available.

## Experiment 13: Rule-based vs Ollama LLM text transformation on a real Japanese WAV fixture

H: A local Japanese LLM through Ollama will follow the text transformation instruction more faithfully than the rule-based backends, while remaining fast enough to use inline in the pipeline.

T: Use the same real Japanese WAV fixture (`outputs/irodori_output.wav`) and the same transform instruction, then compare `rule`, `prompted_rule`, and `ollama` using `hf.co/LiquidAI/LFM2.5-1.2B-JP-GGUF:Q4_K_M` as the local model. Measure transform latency and inspect the transformed text and cached pipeline behavior.

D:
- PASS if the Ollama backend returns a better prompt-following transform than the rule backends and stays within a practical latency budget.
- FAIL if it cannot run locally or is too slow to use inline.
- UNCERTAIN if it is faster but the output quality is not clearly better than the rule baseline.

C:
- The comparison depends on the quality of the STT transcript from the WAV fixture.
- The cache can hide the warm-path cost after the first run, so the cold run matters most for real usage.

U:
- The output-quality judgment is partly subjective, but the result should still be grounded in the concrete transformed strings and latency numbers.

Result:
- `rule` on direct text input returned `[desu-masu] 明日までに資料を送ってください。` essentially instantly.
- `prompted_rule` on direct text input returned the same output and latency was also effectively zero.
- `ollama` with `hf.co/LiquidAI/LFM2.5-1.2B-JP-GGUF:Q4_K_M` returned `明日までに資料を送ってください。` and took about `1.10s` cold / `0.12-0.13s` warm on direct text input.
- On a real WAV fixture, the pipeline source transcript was `Yousmasu! はい、こんにちは!`; `ollama` normalized the punctuation to `Yousmasu! はい、こんにちは！`, while the rule backends kept the artificial `[desu-masu]` prefix.
- Decision: keep `ollama` as the only text-transform backend that actually follows the instruction in a useful way on this machine; keep `rule` / `prompted_rule` only as deterministic fallbacks.

## Experiment 12: `quantumcookie/anime-whisper-ct2*` CT2 smoke attempts on Mac

H: The CT2 Whisper-family repos `quantumcookie/anime-whisper-ct2`, `quantumcookie/anime-whisper-ct2-fp16`, and `quantumcookie/anime-whisper-ct2-int8` will complete a bounded smoke benchmark on this Mac and return a non-empty Japanese transcript.

T: Add dedicated benchmark selectors for the CT2 repos, then run each candidate against the same short Japanese fixture with a bounded wall-clock budget.

D:
- PASS if any candidate completes and returns a non-empty transcript.
- FAIL if none of the candidates complete within a practical benchmark window on this Mac.
- UNCERTAIN if completion occurs but only after a very large startup cost that makes the model impractical for the app.

C:
- The CT2 repositories expose `model.bin`, but the load path may still be too heavy for this machine.
- The benchmark attempts were bounded and did not emit a JSON artifact.

U:
- A different loader or a different machine may get a different result, but for this Mac the practical answer is the one that matters.

Result:
- `quantumcookie/anime-whisper-ct2`: benchmark attempt did not complete in a practical window and produced no JSON artifact.
- `quantumcookie/anime-whisper-ct2-fp16`: benchmark attempt did not complete in a practical window and produced no JSON artifact.
- `quantumcookie/anime-whisper-ct2-int8`: benchmark attempt did not complete in a practical window and produced no JSON artifact.
- Decision: reject CT2 Whisper-family variants for this Mac; stop spending time on this path.

Result:
- `speaker_kv_scale=0.5`, `speaker_kv_min_t=0.5`, `speaker_kv_max_layers=8`: median warm total `0.7090s`, median warm RTF `0.1906`, output duration `3.72s`
- Decision: no meaningful latency improvement versus the existing `num_steps=6 linear` baseline, so keep these as optional quality knobs rather than default changes.

## Experiment 12: Irodori TTS `sway_coeff` sweep at `num_steps=6 sway`

H: If `t_schedule_mode=sway` is used, tuning `sway_coeff` may recover some of the latency lost relative to `linear`.

T: Run the same Japanese text and reference audio with `num_steps=6`, `t_schedule_mode=sway`, and `sway_coeff=0.5`, then compare median warm total time to the existing `num_steps=6 linear` baseline.

D:
- PASS if the warm median improves materially and output remains non-empty.
- FAIL if the runtime breaks.
- UNCERTAIN if the difference is tiny.

C:
- The result is only about latency on one sentence and one reference voice.
- Any perceived audio quality change is not captured here.

U:
- `sway` may primarily affect subjective quality rather than runtime.

Result:
- `num_steps=6`, `t_schedule_mode=sway`, `sway_coeff=0.5`: median warm total `0.7103s`, median warm RTF `0.1909`, output duration `3.72s`
- Decision: no meaningful latency improvement, so `linear` remains the default and `sway` stays an optional alternative.
