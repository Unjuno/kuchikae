# Kuchikae v0.1 Product Specification

## 1. Product definition

Kuchikae is a prompt-conditioned, voice-conditioned speech transformation prototype.

The user records or uploads a short Japanese utterance. The application uses the same audio as:

1. the utterance to transcribe, and
2. the reference audio for voice-conditioned output.

The transcript is transformed according to a free-form `TextTransformPrompt`. The transformed text is then rendered as speech through `VoiceOutputBackend` using `VoiceContext` and a free-form `VoiceOutputPrompt`.

## 2. Core sentence

> Speak once. Kuchikae says it back in your voice, following your prompt.

Japanese concept:

> 話した内容を、プロンプト通りに、自分の声っぽく言い直す。

## 3. v0.1 user flow

1. User records or uploads short Japanese audio.
2. User writes a `TextTransformPrompt`.
3. User writes a `VoiceOutputPrompt`.
4. User runs Kuchikae.
5. App displays source transcript.
6. App displays transformed text.
7. App displays voice context readiness.
8. App plays generated audio.
9. App displays latency report.

## 4. Required user-facing controls

### 4.1 Audio input

- Accept microphone input or uploaded audio file.
- Recommended utterance length: 3 to 10 seconds.
- Maximum v0.1 utterance length: 15 seconds.
- Input language: Japanese.

### 4.2 Text Transform Prompt

A free-form prompt controlling how the transcript is rewritten.

Example:

```text
内容、数字、日時、固有名詞、否定条件は保ちつつ、相手を傷つけない柔らかい言い方に変換してください。
```

This is not a style dropdown. Presets may be added later, but the primary v0.1 control is free-form prompt input.

### 4.3 Voice Output Prompt

A free-form prompt controlling how the output voice should sound.

Example:

```text
元の話者の声質に近づけ、落ち着いた自然な声で、最後に少しニヤつくようなニュアンスを入れてください。
```

### 4.4 Run button

Button label may be:

```text
Kuchikae
```

or:

```text
言い直す
```

The app title remains `Kuchikae Demo`.

## 5. Required outputs

The UI must display:

| Output | Meaning |
|---|---|
| Source Transcript | STT result from input audio |
| Transformed Text | Result of text transformation |
| Output Audio | Generated WAV/audio output |
| Voice Context Status | Whether reference audio is ready |
| Latency Report | STT, text transform, voice output, and total time |

## 6. Explicit non-goals for v0.1

Do not implement:

- fixed style dropdown as the primary control
- `polite/casual/rpg` as hard-coded primary UI choices
- translation
- real-time streaming
- user accounts
- database
- sharing
- multi-speaker management
- mobile-specific UI
- real model integration in the first scaffold
- heavy model dependencies in the first scaffold

## 7. Important conceptual separation

Kuchikae separates text transformation from voice output.

| Layer | Controlled by | Example |
|---|---|---|
| Text transformation | `TextTransformPrompt` | Make it sarcastic, polite, dramatic, news-like |
| Voice output | `VoiceOutputPrompt` | Speak calmly, smile slightly at the end, sound natural |

These must not be merged.

## 8. Default prompts

### 8.1 Default Text Transform Prompt

```text
内容、数字、日時、固有名詞、否定条件は保ちつつ、ユーザーの指示した言い方に変換してください。
```

### 8.2 Default Voice Output Prompt

```text
元の話者の声質に近づけ、自然な声で話してください。
```

## 9. v0.1 success criteria

v0.1 succeeds if:

1. The app can run with dummy backends.
2. The app has the final interface shape for prompt-conditioned and voice-conditioned output.
3. The UI uses prompt inputs, not a fixed style dropdown.
4. The pipeline returns transcript, transformed text, output audio, voice readiness, and latency.
5. The project can be developed through Nix + uv.

v0.1 does not need real STT or real voice cloning in the first implementation. The scaffold must, however, make those backends easy to add later.
