# Backend Contracts

## STTBackend

- Must return non-empty text for readable input or raise a typed error.
- Streaming variants must preserve committed text and keep unstable text isolated.

## TextTransformBackend

- Must return a string.
- Must preserve numbers, negation cues, and named entities when the prompt requests preservation.
- Must not add explanation prefixes to production outputs.

## VoiceOutputBackend

- Must return a readable audio file path.
- The returned audio must be loadable by `soundfile`.
- Streaming output must preserve append-only semantics.

## Quality helpers

- Must expose deterministic helpers for preservation checks.
- Must not depend on model runtime state.
