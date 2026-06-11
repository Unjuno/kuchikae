# Performance Plan

## Goals
1. **Measure repeated same-audio prompt runs** - Track performance improvement when using same audio with different prompts
2. **Avoid redundant STT** - STT results should be cached for the same audio across prompt iterations
3. **Avoid redundant VoiceContext extraction** - VoiceContext should be extracted once per audio and reused
4. **Add chunking scaffold** - Prepare for segmented STT (chunk audio into fixed-size windows)
5. **Optionally add segmented STT later** - Implement `SegmentedSTTBackend` when stable
6. **Optionally add progressive Gradio output later** - Stream transcript, transformed text, then audio
7. **Push-to-talk streaming** - Live partial transcripts during/after recording for real-time feel

## Current Behavior
- `KuchikaePipeline` has `ProcessingCache` for STT and VoiceContext using `AudioKey` (path + size + mtime)
- `process_stream()` yields progressive UI updates: STT → TXT → VOX → DONE
- `process_stream_live()` yields partial STT results: STT_PARTIAL → STT → TXT → VOX → DONE
- `StreamingFasterWhisperSTTBackend` processes audio in 5s chunks with 1s overlap
- `SegmentedSTTBackend` for long audio (>30s) with `FixedWindowSegmenter`
- `PromptedRuleTextTransformBackend` (default) is fast (~ms) with rule-based conversion

## Implementation Priority
1. **Phase 4A**: Add AudioKey and ProcessingCache for better performance tracking ✅
2. **Phase 4B**: Update pipeline to use ProcessingCache for transcript and voice context ✅
3. **Phase 5**: Enable chunking scaffold in `FixedWindowSegmenter` ✅
4. **Phase 6**: Add generator-based streaming (`process_stream`, `process_stream_live`) ✅
5. **Phase 7**: Push-to-talk optimization with `StreamingFasterWhisperSTTBackend` ✅
6. **Phase 8**: Clean up and prepare for release

## Success Metrics
- Same audio + 3 different text prompts: STT called once, VoiceContext extracted once
- ProcessingCache keys must include all relevant prompt fields
- No redundant work across prompt iterations
- Chunking tests for various audio lengths and overlaps
- Progressive output shows transcript → transformed text → audio in order
- Live STT partials update UI every ~5s chunk during transcription

## Performance Tests
```python
# Test repeated same-audio runs
def test_same_audio_different_prompts():
    pipeline = create_pipeline()
    audio_path = "test.wav"
    
    prompt1 = TextTransformPrompt("prompt 1")
    prompt2 = TextTransformPrompt("prompt 2")
    
    result1 = pipeline.process(audio_path, prompt1)
    result2 = pipeline.process(audio_path, prompt2)
    
    # STT should be cached (called once)
    # VoiceContext extraction should be cached (called once)
    # TextTransform should run twice (prompts differ)
    # VoiceOutput should run twice (transformed text differs)

# Test live streaming STT partials
def test_streaming_stt_partials():
    pipeline = create_pipeline({"streaming_stt": True})
    audio_path = "test.wav"
    
    partials = []
    for status, src, txt, aud in pipeline.process_stream_live(audio_path, prompt):
        if status == "STT_PARTIAL":
            partials.append(src)
    
    # Should receive multiple partial updates
    assert len(partials) > 1
```

## Configuration Options
```python
# Enable streaming STT (push-to-talk style)
config = {
    "stt_backend": "faster_whisper",
    "streaming_stt": True,  # Uses StreamingFasterWhisperSTTBackend
}

# Enable segmented STT for long audio
config = {
    "stt_backend": "faster_whisper",
    "segmented_stt": True,  # Uses SegmentedSTTBackend with FixedWindowSegmenter
}
```
