# Kuchikae v0.1 Product Specification

## 1. Product definition

Kuchikae is a prompt-conditioned, voice-conditioned speech transformation prototype.

The user records a short Japanese utterance. The application uses the same audio as:

1. the utterance to transcribe, and
2. the reference audio for voice-conditioned output.

The transcript is transformed according to a text transformation prompt (free-form or template-based). The transformed text is then rendered as speech through `VoiceOutputBackend` using `VoiceContext` and an internally generated `VoiceOutputPrompt`.

## 2. Core sentence

> Speak once. Kuchikae says it back in your voice, following your prompt.

Japanese concept:

> 話した内容を、プロンプト通りに、自分の声っぽく言い直す。

## 3. v0.1 user flow

### 3.1 Normal mode (通常)

1. User records or uploads short Japanese audio.
2. User selects a template from the dropdown, or writes a free-form prompt.
3. User clicks "言い直す".
4. App displays source transcript.
5. App displays transformed text.
6. App shows the applied voice impression label (声の印象).
7. App plays generated audio.

### 3.2 Simple mode (簡易)

1. User selects a template from the dropdown.
2. User presses and holds the PTT button, speaks, then releases.
3. App automatically detects voice emotion and generates voice prompt internally.
4. App displays source transcript and transformed text.
5. App plays generated audio.

## 4. Required user-facing controls

### 4.1 Audio input

- Accept microphone input or uploaded audio file (Normal mode).
- PTT button for push-to-talk recording (Simple mode).
- Recommended utterance length: 3 to 10 seconds.
- Maximum v0.1 utterance length: 15 seconds.
- Input language: Japanese.

### 4.2 Template dropdown

Both Normal and Simple modes provide a template dropdown with the following categories:

| Category | Prefix | Description |
|---|---|---|
| 通常テンプレート | (none) | Natural, polite, casual, short, strong, calm |
| 正式追加テンプレート | (none) | Teacher, friend, news caster, sales, poetic |
| パフォーマンステンプレート | (none) | Announcer, movie trailer, AI assistant, butler, demon king, late-night radio |
| 実験テンプレート | `実験:` | Kansai dialect, gyaru, baby, samurai, sharp-tongued, sarcastic, foreign, specific character |
| 実験強テンプレート | `実験強:` | Stronger variants of experimental templates |
| カスタム | (none) | Empty template for free-form prompt input |

### 4.3 Text transform prompt

A free-form prompt controlling how the transcript is rewritten.

- In Normal mode: visible textbox, editable by the user.
- In Simple mode: automatically filled by template selection, not directly editable.
- Selecting "カスタム" clears the prompt for manual input.
- The custom prompt textbox always overrides the template selection at runtime.

### 4.4 Voice output prompt (internal)

`VoiceOutputPrompt` is **not** a user-facing UI control. It is internally generated based on:

1. User-selected voice style (Normal mode: auto/natural/calm/bright/slow_clear).
2. Audio emotion analysis results (automatic in both modes).
3. Explicit preset priority over auto-detected emotion.

The UI displays the applied voice impression via the `声の印象` label.

### 4.5 Run button

Normal mode: "言い直す" button.
Simple mode: PTT button ("押して話す").

## 5. Required outputs

The UI must display:

| Output | Meaning |
|---|---|
| Source Transcript | STT result from input audio |
| Transformed Text | Result of text transformation |
| Output Audio | Generated WAV/audio output |
| Voice Impression | Applied voice style label (声の印象) |

## 6. Safety

- Experimental templates (`実験:`, `実験強:`) display a safety warning in the UI.
- Warning text: "実験テンプレートは検証用です。なりすまし、詐欺、脅迫、同意のない声の模倣には使用しないでください。"
- Users must not use the tool for impersonation, fraud, harassment, threats, or non-consensual voice imitation.

## 7. Explicit non-goals for v0.1

Do not implement:

- translation
- real-time streaming (beyond STT)
- user accounts
- database
- sharing
- multi-speaker management
- mobile-specific UI
- exact voice cloning guarantee
- heavy model dependencies as required base dependencies

## 8. Important conceptual separation

Kuchikae separates text transformation from voice output.

| Layer | Controlled by | Example |
|---|---|---|
| Text transformation | Template selection or free-form prompt | Make it sarcastic, polite, dramatic, news-like |
| Voice output | Internal (emotion analysis + voice style) | Speak calmly, sound natural, match detected emotion |

These must not be merged into a single control.

## 9. Default prompts

### 9.1 Default Text Transform Prompt

```text
内容、数字、日時、固有名詞、否定条件は保ちつつ、ユーザーの指示した言い方に変換してください。
```

### 9.2 Default Voice Output Prompt (internal)

```text
元の話者の声質に近づけ、自然な声で話してください。
```

## 10. v0.1 success criteria

v0.1 succeeds if:

1. The app can run with dummy backends.
2. The app has Normal and Simple modes with template dropdowns.
3. Normal mode provides a free-form prompt input and voice analysis label.
4. Simple mode provides PTT recording with automatic voice detection.
5. Experimental templates are visible with safety warnings.
6. The pipeline returns transcript, transformed text, output audio, and voice impression.
7. The project can be developed through uv.
