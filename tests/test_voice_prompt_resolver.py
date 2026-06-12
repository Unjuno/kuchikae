"""Tests for VoicePromptResolver."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from kuchikae.domain.audio_emotion import AudioEmotion, DummyAudioEmotionDetector, DisabledAudioEmotionDetector
from kuchikae.domain.voice_style import (
    VoiceMood,
    VoiceSpeed,
    VoiceEmphasis,
    VoiceStyle,
    RuleVoiceStyleDetector,
)
from kuchikae.domain.types import VoiceOutputPrompt
from kuchikae.pipeline.voice_prompt_resolver import VoicePromptResolver


class MockVoiceStyleDetector:
    def __init__(self, return_style: VoiceStyle | None = None):
        self._return_style = return_style or VoiceStyle()
        self.calls: list[str] = []

    def detect(self, text: str) -> VoiceStyle:
        self.calls.append(text)
        return self._return_style


class MockAudioEmotionDetector:
    def __init__(self, return_emotion: AudioEmotion | None = None, disabled: bool = False):
        self._return_emotion = return_emotion or AudioEmotion()
        self.disabled = disabled
        self.calls: list[str] = []

    def detect(self, audio_path: str) -> AudioEmotion:
        self.calls.append(audio_path)
        return self._return_emotion


def test_voice_prompt_resolver_explicit_prompt_priority() -> None:
    """Explicit voice output prompt should be returned as-is."""
    resolver = VoicePromptResolver()
    explicit = VoiceOutputPrompt(instruction="custom prompt")
    result = resolver.build_prompt("source", "transformed", explicit, None)
    assert result == explicit
    assert resolver.last_voice_style == "custom"


def test_voice_prompt_resolver_disabled_audio_emotion_not_started() -> None:
    """Disabled audio emotion detector should not start."""
    detector = DisabledAudioEmotionDetector()
    resolver = VoicePromptResolver(audio_emotion_detector=detector)
    future, executor = resolver.start_audio_emotion("/tmp/test.wav")
    assert future is None
    assert executor is None


def test_voice_prompt_resolver_dummy_detector_emits_events() -> None:
    """Dummy detector should emit start/done events."""
    detector = DummyAudioEmotionDetector()
    emit_calls = []
    
    def mock_emit(name, message, stage, level=None, backend=None, cache=None, elapsed_sec=None, data=None):
        emit_calls.append(name)
    
    resolver = VoicePromptResolver(audio_emotion_detector=detector, emit=mock_emit)
    future, executor = resolver.start_audio_emotion("/tmp/test.wav")
    assert future is not None
    assert executor is not None
    
    emotion = resolver.collect_audio_emotion(future, executor)
    assert emotion is not None
    assert "audio_emotion.detect.start" in emit_calls
    assert "audio_emotion.detect.done" in emit_calls


def test_voice_prompt_resolver_timeout_still_generates_prompt() -> None:
    """Timeout should not prevent prompt generation."""
    detector = MockAudioEmotionDetector(return_emotion=AudioEmotion(mood="happy", confidence=0.9))
    style_detector = MockVoiceStyleDetector(return_style=VoiceStyle(mood=VoiceMood.BRIGHT))
    
    resolver = VoicePromptResolver(
        voice_style_detector=style_detector,
        audio_emotion_detector=detector,
        timeout_sec=0.0,
    )
    
    future, executor = resolver.start_audio_emotion("/tmp/test.wav")
    emotion = resolver.collect_audio_emotion(future, executor)
    
    result = resolver.build_prompt("source", "transformed", None, emotion)
    assert result is not None
    assert "文章内容は変えないでください" in result.instruction


def test_voice_prompt_resolver_source_transformed_fusion() -> None:
    """Source and transformed styles should be fused."""
    source_style = VoiceStyle(mood=VoiceMood.URGENT, confidence=0.9, source="rule")
    transformed_style = VoiceStyle(mood=VoiceMood.WARM, confidence=0.8, source="rule")
    
    call_count = [0]
    def mock_detect(text: str) -> VoiceStyle:
        call_count[0] += 1
        if call_count[0] == 1:
            return source_style
        return transformed_style
    
    style_detector = MagicMock()
    style_detector.detect = mock_detect
    
    resolver = VoicePromptResolver(voice_style_detector=style_detector)
    result = resolver.build_prompt("source text", "transformed text", None, None)
    
    assert result is not None
    assert "文章内容は変えないでください" in result.instruction


def test_voice_prompt_resolver_diagnostics_events() -> None:
    """Should emit source/transformed/fusion diagnostics."""
    emit_calls = []
    
    def mock_emit(name, message, stage, level=None, backend=None, cache=None, elapsed_sec=None, data=None):
        emit_calls.append(name)
    
    resolver = VoicePromptResolver(emit=mock_emit)
    result = resolver.build_prompt("source", "transformed", None, None)
    
    assert "voice_style.detect.start" in emit_calls
    assert "voice_style.source.detect.done" in emit_calls
    assert "voice_style.transformed.detect.done" in emit_calls
    assert "voice_style.fusion.done" in emit_calls
