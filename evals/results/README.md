# Voice eval results

JSONL files in this directory are snapshot baselines produced by
`evals/run_voice_eval.py`.

`output_audio` paths reference local WAV files under `outputs/`.
Those generated WAV files are intentionally **not** committed to git
(see `.gitignore` entries for `outputs/*.wav`).

To inspect the actual generated audio, re-run:

    uv run python evals/run_voice_eval.py --mode tts-only --backend irodori \
        --out evals/results/voice_eval_irodori_tts_only_baseline.jsonl
