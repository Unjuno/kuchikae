"""Tests for voice prompt generation from emotion analysis."""

from __future__ import annotations

import pytest

from kuchikae.domain.voice_prompt import (
    EMOTION_VOICE_PROMPTS,
    build_voice_output_prompt_from_analysis,
    get_emotion_description,
    get_voice_style_display,
)
from kuchikae.domain.types import VoiceOutputPrompt


class TestEmotionVoicePromptsMapping:
    """Test EMOTION_VOICE_PROMPTS mapping contains expected emotions."""

    def test_happy_prompt_exists(self):
        assert "happy" in EMOTION_VOICE_PROMPTS
        assert "明るく前向き" in EMOTION_VOICE_PROMPTS["happy"]

    def test_calm_prompt_exists(self):
        assert "calm" in EMOTION_VOICE_PROMPTS
        assert "落ち着いた穏やか" in EMOTION_VOICE_PROMPTS["calm"]

    def test_sad_prompt_exists(self):
        assert "sad" in EMOTION_VOICE_PROMPTS
        assert "過度に暗くしすぎない" in EMOTION_VOICE_PROMPTS["sad"]

    def test_anger_prompt_exists(self):
        assert "anger" in EMOTION_VOICE_PROMPTS
        assert "攻撃的・威圧的になりすぎない" in EMOTION_VOICE_PROMPTS["anger"]

    def test_neutral_prompt_exists(self):
        assert "neutral" in EMOTION_VOICE_PROMPTS
        assert "自然で聞き取りやすい" in EMOTION_VOICE_PROMPTS["neutral"]

    def test_all_prompts_have_source_preservation(self):
        for key, prompt in EMOTION_VOICE_PROMPTS.items():
            assert "元話者の声質はできるだけ保つ" in prompt, f"{key} prompt missing source preservation"


class TestBuildVoiceOutputPromptFromAnalysis:
    """Test build_voice_output_prompt_from_analysis priority logic."""

    def test_explicit_preset_overrides_emotion(self):
        result = build_voice_output_prompt_from_analysis(emotion="happy", voice_style="calm")
        assert result is not None
        assert "落ち着いた声で" in result.instruction

    def test_auto_with_happy_emotion(self):
        result = build_voice_output_prompt_from_analysis(emotion="happy", voice_style="auto")
        assert result is not None
        assert "明るく前向き" in result.instruction

    def test_auto_with_calm_emotion(self):
        result = build_voice_output_prompt_from_analysis(emotion="calm", voice_style="auto")
        assert result is not None
        assert "落ち着いた穏やか" in result.instruction

    def test_auto_with_sad_emotion(self):
        result = build_voice_output_prompt_from_analysis(emotion="sad", voice_style="auto")
        assert result is not None
        assert "静かで落ち着いた" in result.instruction

    def test_auto_with_anger_emotion(self):
        result = build_voice_output_prompt_from_analysis(emotion="anger", voice_style="auto")
        assert result is not None
        assert "攻撃的・威圧的になりすぎない" in result.instruction

    def test_auto_with_neutral_emotion(self):
        result = build_voice_output_prompt_from_analysis(emotion="neutral", voice_style="auto")
        assert result is not None
        assert "自然で聞き取りやすい" in result.instruction

    def test_auto_with_none_emotion_uses_neutral(self):
        result = build_voice_output_prompt_from_analysis(emotion=None, voice_style="auto")
        assert result is not None
        assert "自然で聞き取りやすい" in result.instruction

    def test_auto_with_unknown_emotion_uses_neutral(self):
        result = build_voice_output_prompt_from_analysis(emotion="unknown_emotion", voice_style="auto")
        assert result is not None
        assert "自然で聞き取りやすい" in result.instruction

    def test_fuzzy_happy_detection(self):
        result = build_voice_output_prompt_from_analysis(emotion="very_happy", voice_style="auto")
        assert result is not None
        assert "明るく前向き" in result.instruction

    def test_fuzzy_anger_detection(self):
        result = build_voice_output_prompt_from_analysis(emotion="angry", voice_style="auto")
        assert result is not None
        assert "攻撃的・威圧的になりすぎない" in result.instruction

    def test_unknown_voice_style_uses_neutral(self):
        result = build_voice_output_prompt_from_analysis(emotion="happy", voice_style="unknown")
        assert result is not None
        assert "自然で聞き取りやすい" in result.instruction

    def test_empty_voice_style_uses_emotion(self):
        result = build_voice_output_prompt_from_analysis(emotion="happy", voice_style="")
        assert result is not None
        assert "明るく前向き" in result.instruction

    def test_all_presets_work(self):
        for preset in ("natural", "calm", "bright", "slow_clear"):
            result = build_voice_output_prompt_from_analysis(emotion="happy", voice_style=preset)
            assert result is not None
            assert isinstance(result, VoiceOutputPrompt)


class TestGetEmotionDescription:
    """Test get_emotion_description returns correct descriptions."""

    def test_happy_description(self):
        desc = get_emotion_description("happy")
        assert desc == "明るく前向きな印象"

    def test_calm_description(self):
        desc = get_emotion_description("calm")
        assert desc == "落ち着いた穏やかな印象"

    def test_sad_description(self):
        desc = get_emotion_description("sad")
        assert desc == "静かで落ち着いた印象"

    def test_anger_description(self):
        desc = get_emotion_description("anger")
        assert desc == "聞き取りやすく制御された印象"

    def test_neutral_description(self):
        desc = get_emotion_description("neutral")
        assert desc == "自然で聞き取りやすい印象"

    def test_none_emotion_description(self):
        desc = get_emotion_description(None)
        assert desc == "自然で聞き取りやすい印象"

    def test_unknown_emotion_description(self):
        desc = get_emotion_description("unknown")
        assert desc == "自然で聞き取りやすい印象"


class TestGetVoiceStyleDisplay:
    """Test get_voice_style_display returns correct display info."""

    def test_preset_displays_custom(self):
        emotion, desc, style = get_voice_style_display("calm", "happy")
        assert emotion == "カスタム設定"
        assert style == "calm"

    def test_auto_displays_emotion(self):
        emotion, desc, style = get_voice_style_display("auto", "happy")
        assert "happy" in emotion
        assert style == "auto"

    def test_auto_with_none_emotion(self):
        emotion, desc, style = get_voice_style_display("auto", None)
        assert "neutral" in emotion
        assert style == "auto"
