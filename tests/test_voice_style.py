from __future__ import annotations

from kuchikae.domain.audio_emotion import AudioEmotion, DummyAudioEmotionDetector
from kuchikae.domain.voice_style import (
    RuleVoiceStyleDetector,
    VoiceMood,
    VoiceSpeed,
    VoiceEmphasis,
    VoiceStyle,
    merge_voice_style,
    voice_style_to_prompt,
)


def test_rule_voice_style_detector_urgent() -> None:
    style = RuleVoiceStyleDetector().detect("至急でお願いします")
    assert style.mood == VoiceMood.URGENT
    assert style.speed == VoiceSpeed.FAST
    assert style.emphasis == VoiceEmphasis.HIGH


def test_rule_voice_style_detector_apologetic() -> None:
    style = RuleVoiceStyleDetector().detect("申し訳ありません")
    assert style.mood == VoiceMood.APOLOGETIC
    assert style.speed == VoiceSpeed.SLOW


def test_rule_voice_style_detector_warm() -> None:
    style = RuleVoiceStyleDetector().detect("ありがとうございます")
    assert style.mood == VoiceMood.WARM


def test_rule_voice_style_detector_serious() -> None:
    style = RuleVoiceStyleDetector().detect("会議の報告です")
    assert style.mood == VoiceMood.SERIOUS


def test_voice_style_prompt_contains_required_phrase() -> None:
    prompt = voice_style_to_prompt(VoiceStyle())
    assert "文章内容は変えないでください" in prompt


def test_dummy_audio_emotion_detector() -> None:
    emotion = DummyAudioEmotionDetector().detect("/tmp/ignored.wav")
    assert emotion.mood == "neutral"
    assert emotion.energy == "medium"
    assert emotion.confidence == 0.0


def test_merge_voice_style_ignores_low_confidence_audio() -> None:
    text_style = VoiceStyle(mood=VoiceMood.URGENT, speed=VoiceSpeed.FAST, emphasis=VoiceEmphasis.HIGH)
    merged = merge_voice_style(text_style, AudioEmotion(confidence=0.1))
    assert merged == text_style

