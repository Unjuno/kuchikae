from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from kuchikae.domain.audio_emotion import AudioEmotion


class VoiceMood(str, Enum):
    NEUTRAL = "neutral"
    CALM = "calm"
    BRIGHT = "bright"
    SERIOUS = "serious"
    APOLOGETIC = "apologetic"
    URGENT = "urgent"
    WARM = "warm"


class VoiceSpeed(str, Enum):
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"


class VoiceClarity(str, Enum):
    NORMAL = "normal"
    HIGH = "high"


class VoiceEmphasis(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class VoiceStyle:
    mood: VoiceMood = VoiceMood.NEUTRAL
    speed: VoiceSpeed = VoiceSpeed.NORMAL
    clarity: VoiceClarity = VoiceClarity.HIGH
    emphasis: VoiceEmphasis = VoiceEmphasis.MEDIUM
    confidence: float = 0.5
    source: str = "default"


@dataclass(frozen=True)
class StyleSignal:
    mood: VoiceMood
    speed: VoiceSpeed
    clarity: VoiceClarity
    emphasis: VoiceEmphasis
    confidence: float
    source: str
    weight: float


class VoiceStyleDetector(Protocol):
    def detect(self, text: str) -> VoiceStyle:
        ...


class RuleVoiceStyleDetector:
    def detect(self, text: str) -> VoiceStyle:
        t = text or ""
        if any(k in t for k in ("至急", "急ぎ", "すぐ", "本日中", "締切", "期限", "緊急")):
            return VoiceStyle(mood=VoiceMood.URGENT, speed=VoiceSpeed.FAST, emphasis=VoiceEmphasis.HIGH, confidence=0.9, source="rule")
        if any(k in t for k in ("申し訳", "すみません", "ご迷惑", "お詫び", "失礼")):
            return VoiceStyle(mood=VoiceMood.APOLOGETIC, speed=VoiceSpeed.SLOW, emphasis=VoiceEmphasis.MEDIUM, confidence=0.85, source="rule")
        if any(k in t for k in ("ありがとう", "助かります", "嬉しい", "よかった", "楽しみ")):
            return VoiceStyle(mood=VoiceMood.WARM, speed=VoiceSpeed.NORMAL, emphasis=VoiceEmphasis.MEDIUM, confidence=0.8, source="rule")
        if any(k in t for k in ("重要", "確認", "報告", "決定", "会議", "契約", "請求", "納期")):
            return VoiceStyle(mood=VoiceMood.SERIOUS, speed=VoiceSpeed.NORMAL, emphasis=VoiceEmphasis.MEDIUM, confidence=0.75, source="rule")
        return VoiceStyle(source="rule")


def audio_emotion_to_voice_style(emotion: AudioEmotion) -> StyleSignal:
    mood = VoiceMood.NEUTRAL
    speed = VoiceSpeed.NORMAL
    emphasis = VoiceEmphasis.MEDIUM

    label = (emotion.mood or "").lower()
    if any(k in label for k in ("happy", "joy", "excited")):
        mood = VoiceMood.BRIGHT
        speed = VoiceSpeed.FAST
        emphasis = VoiceEmphasis.HIGH
    elif any(k in label for k in ("anger", "angry", "ang")):
        mood = VoiceMood.SERIOUS
        speed = VoiceSpeed.NORMAL
        emphasis = VoiceEmphasis.HIGH
    elif "sad" in label:
        mood = VoiceMood.CALM
        speed = VoiceSpeed.SLOW
        emphasis = VoiceEmphasis.LOW
    elif any(k in label for k in ("neutral", "calm")):
        mood = VoiceMood.NEUTRAL
        speed = VoiceSpeed.NORMAL
        emphasis = VoiceEmphasis.MEDIUM

    return StyleSignal(
        mood=mood,
        speed=speed,
        clarity=VoiceClarity.HIGH,
        emphasis=emphasis,
        confidence=emotion.confidence,
        source=f"audio:{emotion.source}",
        weight=0.20,
    )


_SPEED_ORDER = {VoiceSpeed.SLOW: 0, VoiceSpeed.NORMAL: 1, VoiceSpeed.FAST: 2}
_EMPHASIS_ORDER = {VoiceEmphasis.LOW: 0, VoiceEmphasis.MEDIUM: 1, VoiceEmphasis.HIGH: 2}


def _clamp_speed(s: VoiceSpeed) -> VoiceSpeed:
    v = max(0, min(2, _SPEED_ORDER[s]))
    for k, val in _SPEED_ORDER.items():
        if val == v:
            return k
    return VoiceSpeed.NORMAL


def _clamp_emphasis(e: VoiceEmphasis) -> VoiceEmphasis:
    v = max(0, min(2, _EMPHASIS_ORDER[e]))
    for k, val in _EMPHASIS_ORDER.items():
        if val == v:
            return k
    return VoiceEmphasis.MEDIUM


def _shift_speed(s: VoiceSpeed, delta: int) -> VoiceSpeed:
    return _clamp_speed(VoiceSpeed(list(_SPEED_ORDER.keys())[max(0, min(2, _SPEED_ORDER[s] + delta))]))


def _shift_emphasis(e: VoiceEmphasis, delta: int) -> VoiceEmphasis:
    return _clamp_emphasis(VoiceEmphasis(list(_EMPHASIS_ORDER.keys())[max(0, min(2, _EMPHASIS_ORDER[e] + delta))]))


def fuse_voice_styles(
    source_style: VoiceStyle | None,
    transformed_style: VoiceStyle,
    audio_emotion: AudioEmotion | None,
) -> VoiceStyle:
    W_SOURCE = 0.25
    W_TRANSFORMED = 0.50
    W_AUDIO = 0.20

    mood_scores: dict[VoiceMood, float] = {}
    mood_scores[transformed_style.mood] = mood_scores.get(transformed_style.mood, 0) + W_TRANSFORMED
    if source_style is not None:
        mood_scores[source_style.mood] = mood_scores.get(source_style.mood, 0) + W_SOURCE

    audio_signal = None
    if audio_emotion is not None and audio_emotion.confidence >= 0.5:
        audio_signal = audio_emotion_to_voice_style(audio_emotion)
        mood_contrib = min(W_AUDIO, audio_signal.confidence * W_AUDIO)
        mood_scores[audio_signal.mood] = mood_scores.get(audio_signal.mood, 0) + mood_contrib

    final_mood = max(mood_scores, key=lambda m: mood_scores[m]) if mood_scores else VoiceMood.NEUTRAL
    final_confidence = max(mood_scores.values()) if mood_scores else 0.5
    final_confidence = max(0.0, min(1.0, final_confidence))

    final_speed = transformed_style.speed
    final_emphasis = transformed_style.emphasis
    final_clarity = transformed_style.clarity

    if audio_signal is not None and audio_signal.confidence >= 0.5:
        arousal = getattr(audio_emotion, "arousal", 0.5)
        if arousal >= 0.75:
            if final_speed == VoiceSpeed.NORMAL:
                final_speed = VoiceSpeed.FAST
            if final_emphasis in (VoiceEmphasis.LOW, VoiceEmphasis.MEDIUM):
                final_emphasis = _shift_emphasis(final_emphasis, 1)
        elif arousal <= 0.25:
            if final_speed == VoiceSpeed.NORMAL:
                final_speed = VoiceSpeed.SLOW
            if final_emphasis == VoiceEmphasis.HIGH:
                final_emphasis = VoiceEmphasis.MEDIUM

    source_parts = []
    if source_style is not None:
        source_parts.append("source")
    source_parts.append("transformed")
    if audio_signal is not None:
        source_parts.append("audio")
    source_str = "+".join(source_parts)

    return VoiceStyle(
        mood=final_mood,
        speed=final_speed,
        clarity=final_clarity,
        emphasis=final_emphasis,
        confidence=final_confidence,
        source=f"fusion:{source_str}",
    )


def voice_style_to_prompt(style: VoiceStyle) -> str:
    mapping = {
        VoiceMood.NEUTRAL: "自然で聞き取りやすく、",
        VoiceMood.CALM: "落ち着いた声で、",
        VoiceMood.BRIGHT: "明るく、",
        VoiceMood.SERIOUS: "落ち着いて丁寧に、",
        VoiceMood.APOLOGETIC: "申し訳なさが伝わるように、",
        VoiceMood.URGENT: "急ぎの内容として、",
        VoiceMood.WARM: "あたたかく、",
    }
    speed = {VoiceSpeed.SLOW: "少しゆっくり", VoiceSpeed.NORMAL: "自然な速さで", VoiceSpeed.FAST: "やや速めに"}[style.speed]
    clarity = {VoiceClarity.NORMAL: "聞き取りやすく", VoiceClarity.HIGH: "聞き取りやすく明瞭に"}[style.clarity]
    emphasis = {VoiceEmphasis.LOW: "抑揚は控えめに", VoiceEmphasis.MEDIUM: "自然な抑揚で", VoiceEmphasis.HIGH: "はっきりとした抑揚で"}[style.emphasis]
    return f"{mapping[style.mood]}{speed}、{clarity}、{emphasis}読んでください。文章内容は変えないでください。"


def merge_voice_style(text_style: VoiceStyle, audio_emotion: AudioEmotion | None) -> VoiceStyle:
    return fuse_voice_styles(None, text_style, audio_emotion)


VOICE_STYLE_PRESETS = {
    "natural": "自然で聞き取りやすく、元話者の声質に近い雰囲気で読んでください。文章内容は変えないでください。",
    "calm": "落ち着いた声で、自然な速さで、聞き取りやすく読んでください。文章内容は変えないでください。",
    "bright": "明るく、自然な抑揚で、聞き取りやすく読んでください。文章内容は変えないでください。",
    "slow_clear": "少しゆっくり、明瞭に、聞き取りやすく読んでください。文章内容は変えないでください。",
}
