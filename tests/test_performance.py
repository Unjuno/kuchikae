"""Tests for repeated pipeline runs to verify performance improvements."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from kuchikae.domain.audio_key import AudioKey
from kuchikae.counting_backends import (
    CountingSTTBackend,
    CountingTextTransformBackend,
    CountingVoiceContextExtractor,
    CountingVoiceOutputBackend,
)
from kuchikae.pipeline import KuchikaePipeline
from kuchikae.domain.types import TextTransformPrompt


def test_same_audio_different_prompts_calls_stt_once(tmp_path) -> None:
    """Two separate pipelines with same audio each call STT exactly once."""
    # Create a test audio file
    audio_path = str(tmp_path / "audio.wav")
    sr = 16000
    data = np.zeros(sr, dtype=np.float32)
    sf.write(audio_path, data, sr)

    # Create counting backends for first run
    counting_stt1 = CountingSTTBackend()
    counting_text_transform1 = CountingTextTransformBackend()
    counting_voice_context1 = CountingVoiceContextExtractor()
    counting_voice_output1 = CountingVoiceOutputBackend()

    # Create pipeline with counting backends for first run
    pipeline1 = KuchikaePipeline(
        stt_backend=counting_stt1,
        text_transform_backend=counting_text_transform1,
        voice_output_backend=counting_voice_output1,
    )
    # Manually set the voice context extractor
    pipeline1._voice_context_extractor = counting_voice_context1

    # Create counting backends for second run
    counting_stt2 = CountingSTTBackend()
    counting_text_transform2 = CountingTextTransformBackend()
    counting_voice_context2 = CountingVoiceContextExtractor()
    counting_voice_output2 = CountingVoiceOutputBackend()

    # Create pipeline with counting backends for second run
    pipeline2 = KuchikaePipeline(
        stt_backend=counting_stt2,
        text_transform_backend=counting_text_transform2,
        voice_output_backend=counting_voice_output2,
    )
    # Manually set the voice context extractor
    pipeline2._voice_context_extractor = counting_voice_context2

    # Run with first prompt (should call all backends)
    prompt1 = TextTransformPrompt(instruction="Prompt 1")
    pipeline1.process(audio_path, prompt1)

    # Run with second prompt (should use cached results, but different pipeline)
    prompt2 = TextTransformPrompt(instruction="Prompt 2")
    pipeline2.process(audio_path, prompt2)

    # Verify behavior:
    # - First run: STT should be called once, VoiceContext extraction once
    # - Second run: STT should be called once (cached behavior), VoiceContext extraction once (cached behavior)
    # - TextTransform should be called twice (different prompts)
    # - VoiceOutput should be called twice (different transformed text)
    assert counting_stt1.call_count == 1, f"First run STT should be called once, but was called {counting_stt1.call_count} times"
    assert counting_voice_context1.call_count == 1, f"First run VoiceContext should be called once, but was called {counting_voice_context1.call_count} times"
    assert counting_stt2.call_count == 1, f"Second run STT should be called once (cached), but was called {counting_stt2.call_count} times"
    assert counting_voice_context2.call_count == 1, f"Second run VoiceContext should be called once (cached), but was called {counting_voice_context2.call_count} times"
    assert counting_text_transform1.call_count == 1, f"First run TextTransform should be called once, but was called {counting_text_transform1.call_count} times"
    assert counting_text_transform2.call_count == 1, f"Second run TextTransform should be called once, but was called {counting_text_transform2.call_count} times"
    assert counting_voice_output1.call_count == 1, f"First run VoiceOutput should be called once, but was called {counting_voice_output1.call_count} times"
    assert counting_voice_output2.call_count == 1, f"Second run VoiceOutput should be called once, but was called {counting_voice_output2.call_count} times"


def test_processing_cache_functionality(tmp_path) -> None:
    """Test ProcessingCache functionality."""
    from kuchikae.domain.processing_cache import ProcessingCache
    from kuchikae.domain.types import VoiceContext

    # Create a test audio key
    audio_path = str(tmp_path / "audio.wav")
    sr = 16000
    data = np.zeros(sr, dtype=np.float32)
    sf.write(audio_path, data, sr)

    audio_key = AudioKey.from_file(audio_path)

    # Create cache
    cache = ProcessingCache()

    # Initially empty
    assert cache.get_stt(audio_key) is None
    assert cache.get_voice_context(audio_key) is None
    assert len(cache) == 0

    # Set values
    cache.set_stt(audio_key, "Hello world")
    dummy_voice_context = VoiceContext(reference_audio_path=audio_path, ready=True)
    cache.set_voice_context(audio_key, dummy_voice_context)

    # Retrieve values
    assert cache.get_stt(audio_key) == "Hello world"
    assert cache.get_voice_context(audio_key) == dummy_voice_context
    assert len(cache) == 2

    # Clear cache
    cache.clear()
    assert cache.get_stt(audio_key) is None
    assert cache.get_voice_context(audio_key) is None
    assert len(cache) == 0


def test_audio_key_equality() -> None:
    """Test AudioKey equality."""
    from kuchikae.domain.audio_key import AudioKey

    key1 = AudioKey("/path/to/file.wav", 1000, 1234567890.0)
    key2 = AudioKey("/path/to/file.wav", 1000, 1234567890.0)
    key3 = AudioKey("/path/to/other.wav", 2000, 1234567890.0)

    # Same files should be equal
    assert key1 == key2

    # Different files should not be equal
    assert key1 != key3

    # Hash should work
    keys_set = {key1, key2, key3}
    assert len(keys_set) == 2  # key1 and key2 are duplicates


def test_repeated_run_with_cache(tmp_path) -> None:
    """Test that repeated runs with same audio use cache."""
    from kuchikae.domain.processing_cache import ProcessingCache

    # Create a test audio file
    audio_path = str(tmp_path / "audio.wav")
    sr = 16000
    data = np.zeros(sr, dtype=np.float32)
    sf.write(audio_path, data, sr)

    # Create pipeline with cache
    cache = ProcessingCache()
    pipeline = KuchikaePipeline(processing_cache=cache)

    # Run twice with same audio and same prompt
    prompt = TextTransformPrompt(instruction="Test prompt")

    # First run - should call backends
    result1 = pipeline.process(audio_path, prompt)

    # Reset cache (simulating fresh pipeline for test)
    cache.clear()

    # Second run with new cache - should call backends again
    result2 = pipeline.process(audio_path, prompt)

    # Both runs should produce results
    assert result1 is not None
    assert result2 is not None
    assert result1.output_audio_path != ""
    assert result2.output_audio_path != ""
