"""Tests for KuchikaePipeline — check_audio errors, cache behavior, streaming yields."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from kuchikae.domain.audio_key import AudioKey
from kuchikae.counting_backends import (
    CountingSTTBackend,
    CountingTextTransformBackend,
    CountingVoiceContextExtractor,
    CountingVoiceOutputBackend,
)
from kuchikae.pipeline import KuchikaePipeline, MAX_AUDIO_DURATION, MAX_FILE_SIZE
from kuchikae.domain.types import TextTransformPrompt, VoiceContext, VoiceOutputPrompt


# ---------------------------------------------------------------------------
# check_audio
# ---------------------------------------------------------------------------

def _write_wav(path: str, duration_sec: float = 0.1, samplerate: int = 16000) -> None:
    data = np.zeros(int(samplerate * duration_sec), dtype=np.float32)
    sf.write(path, data, samplerate)


def test_check_audio_unsupported_extension(tmp_path) -> None:
    path = str(tmp_path / "test.txt")
    path_s = tmp_path / "test.txt"
    path_s.write_text("not audio")
    pipeline = KuchikaePipeline()
    with pytest.raises(ValueError, match="Unsupported format"):
        pipeline.check_audio(str(path_s))


def test_check_audio_nonexistent_file() -> None:
    pipeline = KuchikaePipeline()
    with pytest.raises(FileNotFoundError):
        pipeline.check_audio("/nonexistent/path.wav")


def test_check_audio_file_too_large(tmp_path) -> None:
    path = str(tmp_path / "large.wav")
    _write_wav(path)
    pipeline = KuchikaePipeline()
    with patch("os.path.getsize", return_value=MAX_FILE_SIZE + 1):
        with pytest.raises(ValueError, match="File too large"):
            pipeline.check_audio(path)


def test_check_audio_too_long(tmp_path) -> None:
    path = str(tmp_path / "long.wav")
    _write_wav(path)
    pipeline = KuchikaePipeline()
    with patch("soundfile.info") as mock_info:
        mock_info.return_value.duration = MAX_AUDIO_DURATION + 10
        with pytest.raises(ValueError, match="Audio too long"):
            pipeline.check_audio(path)


def test_check_audio_valid_wav_passes(tmp_path) -> None:
    path = str(tmp_path / "valid.wav")
    _write_wav(path)
    pipeline = KuchikaePipeline()
    pipeline.check_audio(path)


# ---------------------------------------------------------------------------
# _step_voice cache bug (pipeline.py:202 reads get_stt instead of get_voice_context)
# ---------------------------------------------------------------------------

def test_step_voice_does_not_overwrite_text_with_cached_stt(tmp_path) -> None:
    """_step_voice must NOT overwrite transformed text with cached STT result."""
    wav = tmp_path / "test.wav"
    _write_wav(str(wav))
    audio_key = AudioKey.from_file(str(wav))

    counting_vo = CountingVoiceOutputBackend()
    pipeline = KuchikaePipeline(voice_output_backend=counting_vo)

    vc = VoiceContext(reference_audio_path=str(wav), ready=True)

    pipeline.processing_cache.set_stt(audio_key, "CACHED STT RESULT")
    pipeline.processing_cache.set_voice_context(audio_key, vc)

    result = pipeline._step_voice(
        text="TRANSFORMED TEXT",
        audio_path=str(wav),
        audio_key=audio_key,
        voice_context=vc,
    )

    assert pipeline.voice_output_backend == counting_vo
    last_call_text = counting_vo.last_text
    assert last_call_text == "TRANSFORMED TEXT", (
        f"Expected 'TRANSFORMED TEXT', got {last_call_text!r}. "
        "Bug: _step_voice reads get_stt(audio_key) instead of get_voice_context."
    )


def test_step_voice_uses_cached_voice_context(tmp_path) -> None:
    wav = tmp_path / "test.wav"
    _write_wav(str(wav))
    audio_key = AudioKey.from_file(str(wav))

    counting_vc = CountingVoiceContextExtractor()
    pipeline = KuchikaePipeline()
    pipeline._voice_context_extractor = counting_vc

    cached_vc = VoiceContext(reference_audio_path="/cached/path.wav", ready=True)

    pipeline.processing_cache.set_voice_context(audio_key, cached_vc)

    pipeline._step_voice(
        text="hello",
        audio_path=str(wav),
        audio_key=audio_key,
        voice_context=VoiceContext(reference_audio_path="", ready=False),
    )

    assert counting_vc.call_count == 0, "Voice context extractor should NOT be called when cache hit"

    stored_vc = pipeline.processing_cache.get_voice_context(audio_key)
    assert stored_vc is not None
    assert stored_vc.reference_audio_path == "/cached/path.wav"


# ---------------------------------------------------------------------------
# process_stream
# ---------------------------------------------------------------------------

def test_process_stream_yields_correct_sequence(tmp_path) -> None:
    wav = tmp_path / "stream.wav"
    _write_wav(str(wav))
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="テスト")
    results = list(pipeline.process_stream(str(wav), prompt))

    assert len(results) >= 2

    statuses = [r[0] for r in results]
    assert statuses == ["STT", "TXT", "VOX", "DONE"]

    final = results[-1]
    assert final[0] == "DONE"
    assert isinstance(final[1], str) and len(final[1]) > 0
    assert isinstance(final[2], str) and len(final[2]) > 0
    assert isinstance(final[3], str) and len(final[3]) > 0


def test_process_stream_caches_stt_and_voice_context(tmp_path) -> None:
    wav = tmp_path / "cache.wav"
    _write_wav(str(wav))
    audio_key = AudioKey.from_file(str(wav))

    counting_stt = CountingSTTBackend()
    pipeline = KuchikaePipeline(stt_backend=counting_stt)
    prompt = TextTransformPrompt(instruction="テスト")

    list(pipeline.process_stream(str(wav), prompt))

    assert pipeline.processing_cache.get_stt(audio_key) is not None
    assert pipeline.processing_cache.get_voice_context(audio_key) is not None
    assert counting_stt.call_count == 1

    list(pipeline.process_stream(str(wav), prompt))
    assert counting_stt.call_count == 1, "STT should be called only once (cached)"


# ---------------------------------------------------------------------------
# process_stream_live
# ---------------------------------------------------------------------------

def test_process_stream_live_yields_correct_sequence(tmp_path) -> None:
    wav = tmp_path / "live.wav"
    _write_wav(str(wav))
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="テスト")
    results = list(pipeline.process_stream_live(str(wav), prompt))

    assert len(results) >= 2

    statuses = [r[0] for r in results]
    assert "STT" in statuses
    assert "TXT" in statuses
    assert "VOX" in statuses
    assert "DONE" in statuses

    final = results[-1]
    assert final[0] == "DONE"
    assert isinstance(final[1], str) and len(final[1]) > 0
    assert isinstance(final[2], str) and len(final[2]) > 0
    assert isinstance(final[3], str) and len(final[3]) > 0


def test_process_stream_live_uses_cached_stt(tmp_path) -> None:
    wav = tmp_path / "live_cache.wav"
    _write_wav(str(wav))
    audio_key = AudioKey.from_file(str(wav))

    counting_stt = CountingSTTBackend()
    pipeline = KuchikaePipeline(stt_backend=counting_stt)
    prompt = TextTransformPrompt(instruction="テスト")

    list(pipeline.process_stream_live(str(wav), prompt))
    assert counting_stt.call_count == 1

    list(pipeline.process_stream_live(str(wav), prompt))
    assert counting_stt.call_count == 1, "STT should be cached after first run"


def test_process_passes_voice_output_prompt(tmp_path) -> None:
    wav = tmp_path / "voice_prompt.wav"
    _write_wav(str(wav))

    counting_vo = CountingVoiceOutputBackend()
    pipeline = KuchikaePipeline(voice_output_backend=counting_vo)

    text_prompt = TextTransformPrompt(instruction="テスト")
    voice_prompt = VoiceOutputPrompt.from_file()

    pipeline.process(str(wav), text_prompt, voice_prompt)

    assert counting_vo.call_count == 1
    assert counting_vo.last_prompt == voice_prompt


def test_process_stream_passes_voice_output_prompt(tmp_path) -> None:
    wav = tmp_path / "voice_prompt_stream.wav"
    _write_wav(str(wav))

    counting_vo = CountingVoiceOutputBackend()
    pipeline = KuchikaePipeline(voice_output_backend=counting_vo)

    text_prompt = TextTransformPrompt(instruction="テスト")
    voice_prompt = VoiceOutputPrompt.from_file()

    list(pipeline.process_stream(str(wav), text_prompt, voice_prompt))

    assert counting_vo.call_count == 1
    assert counting_vo.last_prompt == voice_prompt


def test_process_stream_live_passes_voice_output_prompt(tmp_path) -> None:
    wav = tmp_path / "voice_prompt_stream_live.wav"
    _write_wav(str(wav))

    counting_vo = CountingVoiceOutputBackend()
    pipeline = KuchikaePipeline(voice_output_backend=counting_vo)

    text_prompt = TextTransformPrompt(instruction="テスト")
    voice_prompt = VoiceOutputPrompt.from_file()

    list(pipeline.process_stream_live(str(wav), text_prompt, voice_prompt))

    assert counting_vo.call_count == 1
    assert counting_vo.last_prompt == voice_prompt


# ---------------------------------------------------------------------------
# warmup — error handling should not raise
# ---------------------------------------------------------------------------

def test_warmup_noop_with_dummy_backends(tmp_path) -> None:
    pipeline = KuchikaePipeline()
    pipeline.warmup()


def test_warmup_does_not_raise_on_error(tmp_path) -> None:
    pipeline = KuchikaePipeline()

    with patch.object(pipeline.stt_backend, "transcribe", side_effect=Exception("boom")):
        pipeline.warmup()

    from unittest.mock import MagicMock
    mock_vo = MagicMock(spec=["_ensure_runtime"])
    mock_vo._ensure_runtime.side_effect = Exception("boom")
    old_vo = pipeline.voice_output_backend
    pipeline.voice_output_backend = mock_vo
    pipeline.warmup()
    pipeline.voice_output_backend = old_vo

    with patch("httpx.post", side_effect=Exception("boom")):
        pipeline.warmup()
