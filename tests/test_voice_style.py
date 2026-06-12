from __future__ import annotations

from kuchikae.domain.audio_emotion import AudioEmotion, DummyAudioEmotionDetector
from kuchikae.domain.voice_style import (
    RuleVoiceStyleDetector,
    VoiceMood,
    VoiceSpeed,
    VoiceEmphasis,
    VoiceStyle,
    StyleSignal,
    audio_emotion_to_voice_style,
    fuse_voice_styles,
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
    assert merged.mood == text_style.mood
    assert merged.speed == text_style.speed
    assert merged.emphasis == text_style.emphasis


def test_audio_emotion_to_voice_style_happy() -> None:
    signal = audio_emotion_to_voice_style(AudioEmotion(mood="happy", confidence=0.8))
    assert signal.mood == VoiceMood.BRIGHT
    assert signal.speed == VoiceSpeed.FAST
    assert signal.emphasis == VoiceEmphasis.HIGH
    assert signal.confidence == 0.8


def test_audio_emotion_to_voice_style_sad() -> None:
    signal = audio_emotion_to_voice_style(AudioEmotion(mood="sad", confidence=0.7))
    assert signal.mood == VoiceMood.CALM
    assert signal.speed == VoiceSpeed.SLOW
    assert signal.emphasis == VoiceEmphasis.LOW


def test_audio_emotion_to_voice_style_neutral() -> None:
    signal = audio_emotion_to_voice_style(AudioEmotion(mood="neutral", confidence=0.6))
    assert signal.mood == VoiceMood.NEUTRAL
    assert signal.speed == VoiceSpeed.NORMAL
    assert signal.emphasis == VoiceEmphasis.MEDIUM


def test_fuse_voice_styles_transformed_weight_dominates() -> None:
    source = VoiceStyle(mood=VoiceMood.URGENT, speed=VoiceSpeed.FAST, confidence=0.9, source="rule")
    transformed = VoiceStyle(mood=VoiceMood.WARM, speed=VoiceSpeed.NORMAL, confidence=0.8, source="rule")
    result = fuse_voice_styles(source, transformed, None)
    assert result.mood == VoiceMood.WARM
    assert result.source.startswith("fusion:")


def test_fuse_voice_styles_audio_influences_speed() -> None:
    source = VoiceStyle(mood=VoiceMood.NEUTRAL, speed=VoiceSpeed.NORMAL, confidence=0.5, source="rule")
    transformed = VoiceStyle(mood=VoiceMood.NEUTRAL, speed=VoiceSpeed.NORMAL, confidence=0.5, source="rule")
    audio = AudioEmotion(mood="happy", arousal=0.8, confidence=0.7)
    result = fuse_voice_styles(source, transformed, audio)
    assert result.speed == VoiceSpeed.FAST
    assert result.emphasis == VoiceEmphasis.HIGH


def test_fuse_voice_styles_low_confidence_audio_ignored() -> None:
    source = VoiceStyle(mood=VoiceMood.URGENT, confidence=0.9, source="rule")
    transformed = VoiceStyle(mood=VoiceMood.WARM, confidence=0.8, source="rule")
    audio = AudioEmotion(mood="happy", confidence=0.3)
    result = fuse_voice_styles(source, transformed, audio)
    assert result.mood == VoiceMood.WARM


def test_fuse_voice_styles_no_source() -> None:
    transformed = VoiceStyle(mood=VoiceMood.SERIOUS, confidence=0.7, source="rule")
    result = fuse_voice_styles(None, transformed, None)
    assert result.mood == VoiceMood.SERIOUS
    assert result.source == "fusion:transformed"


def test_fuse_voice_styles_all_sources() -> None:
    source = VoiceStyle(mood=VoiceMood.URGENT, confidence=0.9, source="rule")
    transformed = VoiceStyle(mood=VoiceMood.WARM, confidence=0.8, source="rule")
    audio = AudioEmotion(mood="happy", arousal=0.8, confidence=0.7)
    result = fuse_voice_styles(source, transformed, audio)
    assert "source" in result.source
    assert "transformed" in result.source
    assert "audio" in result.source


def test_fuse_voice_styles_confidence_weighted() -> None:
    source = VoiceStyle(mood=VoiceMood.URGENT, confidence=0.3, source="rule")
    transformed = VoiceStyle(mood=VoiceMood.WARM, confidence=0.9, source="rule")
    result = fuse_voice_styles(source, transformed, None)
    assert result.mood == VoiceMood.WARM


def test_fuse_voice_styles_low_confidence_source_ignored() -> None:
    source = VoiceStyle(mood=VoiceMood.URGENT, confidence=0.1, source="rule")
    transformed = VoiceStyle(mood=VoiceMood.WARM, confidence=0.9, source="rule")
    result = fuse_voice_styles(source, transformed, None)
    assert result.mood == VoiceMood.WARM


def test_fuse_voice_styles_high_confidence_source_wins() -> None:
    source = VoiceStyle(mood=VoiceMood.URGENT, confidence=0.95, source="rule")
    transformed = VoiceStyle(mood=VoiceMood.WARM, confidence=0.3, source="rule")
    result = fuse_voice_styles(source, transformed, None)
    assert result.mood == VoiceMood.URGENT


def test_audio_emotion_mood_anger() -> None:
    signal = audio_emotion_to_voice_style(AudioEmotion(mood="anger", confidence=0.7))
    assert signal.mood == VoiceMood.SERIOUS
    assert signal.speed == VoiceSpeed.NORMAL
    assert signal.emphasis == VoiceEmphasis.HIGH


def test_audio_emotion_mood_calm() -> None:
    signal = audio_emotion_to_voice_style(AudioEmotion(mood="calm", confidence=0.6))
    assert signal.mood == VoiceMood.CALM
    assert signal.speed == VoiceSpeed.SLOW
    assert signal.emphasis == VoiceEmphasis.MEDIUM


def test_audio_emotion_mood_neutral() -> None:
    signal = audio_emotion_to_voice_style(AudioEmotion(mood="neutral", confidence=0.5))
    assert signal.mood == VoiceMood.NEUTRAL
    assert signal.speed == VoiceSpeed.NORMAL
    assert signal.emphasis == VoiceEmphasis.MEDIUM

