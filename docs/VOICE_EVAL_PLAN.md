# Voice/audio eval plan

## Phase 1: 現状監査

### VoiceOutputPrompt 作成箇所

`VoiceOutputPrompt` は `kuchikae/domain/types.py:24` で定義された `instruction: str` だけの単純な dataclass。

作成パス（本番）:
1. `pipeline.py:640` `_build_voice_prompt()` → `voice_prompt.py:95` `build_voice_output_prompt_from_analysis()` が `VoiceOutputPrompt` を生成
2. 優先順位: explicit preset → emotion-based → neutral fallback

未使用の並列パス:
- `pipeline/voice_prompt_resolver.py` の `VoicePromptResolver` — text-based style detection + fusion を行うが、本番では使われていない。テストのみ。

### Audio emotion detector の出力

`domain/audio_emotion.py:12` `AudioEmotion` dataclass:
- `mood: str` — "happy" / "anger" / "sad" / "calm" / "neutral"
- `energy: str` — "high" / "medium" / "low"
- `arousal: float` — 0.0–1.0 (興奮度)
- `valence: float` — -0.7–+0.7 (快不快)
- `confidence: float` — 0.0–1.0
- `source: str` — "dummy" / "disabled" / "transformers:model_id"

本番では `DummyAudioEmotionDetector` が使われており、常に `(neutral, medium, 0.5, 0.0, 0.0, "dummy")` を返す。
`TransformersAudioEmotionDetector` (wav2vec2-base-superb-er) は実装済みだが、パイプラインの emotion 検出 timeout が 50ms と短く実質使われていない。

### voice_style / emotion / source audio → TTS backend の伝達

```
UI voice_style Radio (通常タブのみ)
  → handlers.py run(voice_style="auto")
    → pipeline.py process_stream(path, prompt, voice_style)
      → _build_voice_prompt(source, transformed, emotion, voice_style)
        → build_voice_output_prompt_from_analysis()  → VoiceOutputPrompt (instruction文字列)
      → _step_voice(text, voice_context, voice_output_prompt)
        → voice_output_backend.synthesize(text, voice_context, prompt)
```

**重要 (2026-06-15 修正):** `IrodoriTTSVoiceOutputBackend.synthesize()` は `prompt` パラメータを無視していたが、`SamplingRequest(caption=prompt.instruction)` に渡すよう修正済み (voice_output.py:279)。
Irodori-TTS の内部で `model_cfg.use_caption_condition` が True の場合に caption が生成に影響する。`cfg_scale_caption` は `cfg_scale_text` と同じ値が使われており、個別の環境変数はない。

`OpenVoiceOutputBackend.synthesize()` は引き続き prompt を無視しているが、OpenVoice に caption 相当の API は存在しない。

### デッドコード: VoicePromptResolver

`pipeline/voice_prompt_resolver.py` の `VoicePromptResolver` クラスは text-based voice style detection (source → transformed text を解析) + emotion fusion を行うが、本番の pipeline.py からは一切呼ばれていない。テスト (`test_voice_prompt_resolver.py`) のみ存在。
代わりに pipeline.py はインラインの `_detect_audio_emotion_async()` / `_collect_audio_emotion()` / `_build_voice_prompt()` を使用している。

同じく `handlers.py:102` の `resolve_voice_style()` も未使用。`run()` は `voice_style` 文字列を pipeline に渡し、`_build_voice_prompt()` 内で `build_voice_output_prompt_from_analysis()` が変換する。

### normal mode vs simple mode の差異

| 項目 | run() (通常) | run_simple() (簡易) |
|------|-------------|-------------------|
| voice_style 引数 | あり (UI Radio から) | なし (常に pipeline 既定値 "auto") |
| pipeline 呼び出し | stream_fn(path, prompt, voice_style) | stream_fn(path, prompt) |
| voice_analysis 表示 | あり (yield に含む) | なし |
| emotion 検出 | 両モードとも行われる (違いなし) |

simple mode では voice_style を指定できない。template_name のみ選択可能。

### Audio emotion detection timeout

pipeline.py:303 の `voice_style_timeout_sec: float = 0.05` (50ms) が audio emotion 検出の上限。
`DummyAudioEmotionDetector` は即時応答するため問題ないが、`TransformersAudioEmotionDetector` (wav2vec2) が使われる場合、CPU 推論は 50ms を超えるため常にタイムアウト → None → neutral fallback となる。
実質的に emotion 検出を機能させるには 5–10 秒程度に緩和する必要がある。

### Irodori-TTS の reference audio 利用

`synthesize()` (`backends/voice_output.py:252`) は `voice_context.reference_audio_path` を `SamplingRequest(ref_wav=...)` に渡している。
Irodori-TTS 内部で reference audio から話者特徴を抽出している。
つまり **speaker timbre は reference WAV から再構築される**。

しかし VoiceOutputPrompt (emotion/style 指示) は無視されるため、感情や話し方の制御は reference audio の質に完全に依存している。

### 話者情報喪失ポイント

1. **STT 段階** (最大損失): WAV → text で全音響情報が失われる。timbre, pitch, prosody, emotion, speaking rate が全て消える。
2. **VoiceContextExtractor** (`domain/audio_cache.py`): v0.1 では embedding 計算なし。pass-through のみ。
3. **TTS backend の prompt 無視**: emotion/style instruction が捨てられる。
4. **Audio emotion timeout**: 50ms では実モデル推論が間に合わず、dummy に fallback。

---

## Phase 2: voice eval skeleton

以下ファイルを作成済み:
- `evals/voice_cases.yaml` — eval cases
- `evals/run_voice_eval.py` — JSONL 出力 runner
- `evals/summarize_voice_eval.py` — 集計
- `evals/fixtures/voice/.gitkeep` — fixture 用ディレクトリ
- `tests/test_voice_eval_cases.py` — 構造テスト

JSONL schema:
```json
{
  "case_id": "...",
  "input_audio": "...",
  "output_audio": "...",
  "template": "...",
  "input_text": "...",
  "transformed_text": "...",
  "voice_backend": "irodori",
  "speaker_similarity": null,
  "duration_ratio": null,
  "pitch_delta_mean": null,
  "energy_delta_db": null,
  "input_emotion": null,
  "output_emotion": null,
  "verdict": "pass|warn|fail|skip",
  "failure_reason": ""
}
```

---

## Phase 3: 代替 TTS backend 比較

| 項目 | Irodori-TTS (現行) | F5-TTS | CosyVoice / CosyVoice2 | IndexTTS | RVC (post-conversion) | XTTS 系 |
|------|-------------------|--------|------------------------|----------|----------------------|---------|
| **Japanese support** | 日本語特化 (Aratako 製) | 中。英語/中国語中心、日本語は fine-tune が必要 | 中。CosyVoice2 は中国語+英語中心 | 日本語対応 (GitHub 上で Japanese demo) | 言語非依存 (voice conversion) | 中。多言語対応だが日本語品質は限定的 |
| **Zero-shot voice cloning** | 対応。ref_wav から話者特徴抽出 | 対応。prompt audio で clone | CosyVoice2 で対応改善 (3s reference) | 対応 (3-10s reference) | **変換専用**。入力音声の声質をターゲット話者に変換 | 対応 (6s reference) |
| **Reference audio support** | ✅ 対応 (ref_wav 利用) | ✅ 対応 (prompt audio) | ✅ 対応 (reference embedding) | ✅ 対応 (reference) | ✅ 必須 (変換元音声) | ✅ 対応 |
| **Emotion/style control** | ❌ prompt 無視 (制御不能) | ❌ なし (zero-shot のみ) | ❌ なし (CosyVoice2 で改善傾向) | ❌ なし | ❌ なし (voice conversion) | ⚠️ emotion encoder あり (限定的) |
| **Local inference** | ✅ MPS/CUDA/CPU | ✅ (DiT, 要 GPU) | ⚠️ 要 CUDA、Linux 推奨 | ⚠️ 要 GPU | ✅ 軽量 (要 CUDA) | ⚠️ 要 GPU (VITS 系) |
| **macOS feasibility** | ✅ MPS 動作確認済み | ⚠️ MPS 未検証、重い | ❌ Linux のみ実質 | ❌ CUDA 前提 | ⚠️ MPS で一部動作 | ❌ CUDA 前提 |
| **License** | Apache 2.0 | CC BY-NC-SA 4.0 | CosyVoice: Apache 2.0 / CosyVoice2: CC BY-NC-SA 4.0 | CC BY-NC-SA 4.0 | MIT (RVC) | CPML (XTTSv2) / Apache 2.0 (Coqui) |
| **Expected integration difficulty** | 低 (導入済み) | 高 (モデル大、推論設計要) | 高 (依存重い、CUDA 前提) | 中 (Python API あり) | 中 (RVC は別処理、pipeline 設計要) | 中 (TTS として統合可) |
| **備考** | emotion 制御が効かない点が最大の課題 | 音質高いがサイズ大 | 中国語以外の品質未知 | 日本語品質期待できるがやや重い | voice conversion なので別用途 | コミュニティ分散 |

### 推奨優先順位 (今後)
1. **Irodori-TTS 改善** (prompt を caption として渡すよう修正) — ✅ 済み (2026-06-15)
2. **IndexTTS** 調査 — 日本語品質が最も期待できる代替
3. **F5-TTS** 長期検討 — 音質最高だがリソース要求大
