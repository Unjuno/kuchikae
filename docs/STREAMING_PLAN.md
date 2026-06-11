# Streaming / Near-Real-Time Pipeline Plan

## Goal

Extend Kuchikae from batch (record → process → hear) to streaming
(while-recording, incremental STT → incremental transform → incremental TTS).

## Architecture

```
Microphone → AudioChunker → StreamingSTTBackend → IncrementalTextTransformBackend
                                                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ StreamingVoiceOutputBackend ← AudioSegmentQueue ← segment_sentences│
└─────────────────────────────────────────────────────────────────────┘
                                        ↓
                                  Output audio
```

## Key Design Decisions

1. **Latency measured, not assumed** — `StreamingMetricsRecorder` with `perf_counter`
2. **Stable/unstable separation** — STT partials expose `stable_prefix` so downstream only
   consumes committed text
3. **Segment-based TTS output** — text is split into sentence/clause units;
   `AudioSegmentQueue` merges them in order regardless of synthesis order
4. **Incremental transform** — only new committed text is re-processed; state is tracked
   via `TransformState.transformed_up_to`

## Current State (Phase 0 — completed)

- Core types: `AudioChunk` (frozen), `STTPartial`/`STTCommit`/`STTFinal`,
  `TransformState`/`TransformUpdate`, `StreamingAudioSegment`,
  `StreamingLatencyReport`, `StreamChunk`
- Interfaces: `StreamingSTTBackend`, `IncrementalTextTransformBackend`,
  `StreamingVoiceOutputBackend` (all with `Dummy*` implementations)
- Utilities: `AudioChunker`, `EnergyVAD`, `AudioStreamBuffer`, `AudioSegmentQueue`,
  `segment_sentences`/`segment_clauses`
- Metrics: `LatencyLogger` (JSONL), `StreamingMetricsRecorder`
- Quality helpers: `check_numeric_preservation`, `check_negation_preservation`

## Phase 1 — Incremental Model Integration

- Wire real STT (FasterWhisper) into `StreamingSTTBackend.push_audio()` using
  `AudioChunker` output
- Wire real text transform (Ollama/GPT) into `IncrementalTextTransformBackend`
  — currently `DummyIncrementalTextTransformBackend` prefixes text
- Wire real TTS (Irodori/OpenVoice) into `StreamingVoiceOutputBackend`

## Phase 2 — Pipeline Orchestration

- `StreamingPipeline` class that wires all three backends together
- Manages session lifecycle: `start_session()` → `push_audio()` → `flush()`
- Integrates `StreamingMetricsRecorder` to produce `StreamingLatencyReport`

## Phase 3 — UI Integration

- Extend Gradio UI with streaming mode
- Expose incremental partials in UI (text display update as STT yields)
- Audio playback of incremental TTS segments (play-while-synthesizing)

## Performance Targets (Phase 2+)

| Metric | Target | Measurement |
|---|---|---|
| Time to first partial transcript | < 1.5 s | `StreamingLatencyReport.time_to_first_partial_transcript` |
| Time to first audio output | < 4.0 s | `StreamingLatencyReport.time_to_first_audio` |
| Realtime factor | < 1.0 | `StreamingLatencyReport.realtime_factor` |

## Dependency Rules

Phase 1 must not add new heavy ML dependencies (`torch`, `transformers`,
`faster-whisper`, `openai`, `anthropic`, `elevenlabs`, etc.) beyond what
the existing batch pipeline already uses.
