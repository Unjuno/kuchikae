"""Voice prompt generation from audio emotion analysis.

Generates VoiceOutputPrompt from detected emotion, with priority:
1. Explicit voice_style preset (natural, calm, bright, slow_clear)
2. Auto: emotion-based prompt generation
3. Fallback: neutral prompt
"""

from __future__ import annotations

from kuchikae.domain.audio_emotion import AudioEmotion
from kuchikae.domain.types import VoiceOutputPrompt
from kuchikae.domain.voice_style import VOICE_STYLE_PRESETS

# Emotion-based voice prompts for TTS synthesis
EMOTION_VOICE_PROMPTS: dict[str, str] = {
    "happy": (
        "検出された声の印象: happy\n"
        "以下の方針で音声を合成してください。\n"
        "- 元話者の声質はできるだけ保つ。\n"
        "- 明るく前向きな印象を保つ。\n"
        "- 抑揚を少し豊かにする。\n"
        "- 話速は自然または少し軽快にする。\n"
        "- 聞き取りやすく自然な日本語音声にする。"
    ),
    "calm": (
        "検出された声の印象: calm\n"
        "以下の方針で音声を合成してください。\n"
        "- 元話者の声質はできるだけ保つ。\n"
        "- 落ち着いた穏やかな印象を保つ。\n"
        "- 抑揚は控えめにする。\n"
        "- 話速はややゆっくりにする。\n"
        "- 聞き取りやすく自然な日本語音声にする。"
    ),
    "sad": (
        "検出された声の印象: sad\n"
        "以下の方針で音声を合成してください。\n"
        "- 元話者の声質はできるだけ保つ。\n"
        "- 静かで落ち着いた印象を保つ。\n"
        "- 過度に暗くしすぎない。\n"
        "- 抑揚は控えめにし、聞き取りやすさを優先する。\n"
        "- 自然な日本語音声にする。"
    ),
    "anger": (
        "検出された声の印象: anger\n"
        "以下の方針で音声を合成してください。\n"
        "- 元話者の声質はできるだけ保つ。\n"
        "- 強い感情は残しすぎず、聞き取りやすく制御する。\n"
        "- 語尾を明瞭にする。\n"
        "- 攻撃的・威圧的になりすぎないようにする。\n"
        "- 自然な日本語音声にする。"
    ),
    "neutral": (
        "検出された声の印象: neutral\n"
        "以下の方針で音声を合成してください。\n"
        "- 元話者の声質はできるだけ保つ。\n"
        "- 自然で聞き取りやすい発話にする。\n"
        "- 抑揚と話速は標準的にする。\n"
        "- 過度な演出を避ける。"
    ),
}

# Fallback prompt for unknown emotions
_NEUTRAL_PROMPT = EMOTION_VOICE_PROMPTS["neutral"]

# Short description for UI display
EMOTION_DESCRIPTIONS: dict[str, str] = {
    "happy": "明るく前向きな印象",
    "calm": "落ち着いた穏やかな印象",
    "sad": "静かで落ち着いた印象",
    "anger": "聞き取りやすく制御された印象",
    "neutral": "自然で聞き取りやすい印象",
}


def _resolve_emotion_key(emotion: str | None) -> str:
    """Map emotion string to a key in EMOTION_VOICE_PROMPTS."""
    if not emotion:
        return "neutral"
    emotion_lower = emotion.lower().strip()
    if emotion_lower in EMOTION_VOICE_PROMPTS:
        return emotion_lower
    # Fuzzy matching for common variants
    if any(k in emotion_lower for k in ("happy", "joy", "excited")):
        return "happy"
    if any(k in emotion_lower for k in ("anger", "angry", "ang")):
        return "anger"
    if "sad" in emotion_lower:
        return "sad"
    if "calm" in emotion_lower:
        return "calm"
    return "neutral"


def build_voice_output_prompt_from_analysis(
    emotion: str | None = None,
    voice_style: str = "auto",
) -> VoiceOutputPrompt | None:
    """Build VoiceOutputPrompt from audio emotion analysis and voice style selection.

    Priority:
    1. voice_style preset (natural, calm, bright, slow_clear) -> use preset prompt
    2. voice_style == "auto" -> generate from emotion
    3. emotion unknown/None -> use neutral prompt

    Returns VoiceOutputPrompt or None if auto with no emotion data.
    """
    # Priority 1: Explicit preset
    if voice_style and voice_style != "auto" and voice_style in VOICE_STYLE_PRESETS:
        return VoiceOutputPrompt(instruction=VOICE_STYLE_PRESETS[voice_style])

    # Priority 2: Auto mode with emotion
    if voice_style == "auto" or not voice_style:
        emotion_key = _resolve_emotion_key(emotion)
        prompt_text = EMOTION_VOICE_PROMPTS.get(emotion_key, _NEUTRAL_PROMPT)
        return VoiceOutputPrompt(instruction=prompt_text)

    # Priority 3: Unknown voice_style -> neutral
    return VoiceOutputPrompt(instruction=_NEUTRAL_PROMPT)


def get_emotion_description(emotion: str | None) -> str:
    """Get short Japanese description of detected emotion for UI display."""
    if not emotion:
        emotion = "neutral"
    emotion_key = _resolve_emotion_key(emotion)
    return EMOTION_DESCRIPTIONS.get(emotion_key, EMOTION_DESCRIPTIONS["neutral"])


def get_voice_style_display(
    voice_style: str,
    emotion: str | None,
) -> tuple[str, str, str]:
    """Get display information for voice analysis label.

    Returns:
        (emotion_label, prompt_description, applied_style)
    """
    if voice_style and voice_style != "auto" and voice_style in VOICE_STYLE_PRESETS:
        return (
            "カスタム設定",
            "ユーザー指定のスタイル",
            voice_style,
        )

    emotion_key = _resolve_emotion_key(emotion)
    description = EMOTION_DESCRIPTIONS.get(emotion_key, EMOTION_DESCRIPTIONS["neutral"])
    return (
        f"声の印象: {emotion_key}",
        description,
        "auto",
    )
