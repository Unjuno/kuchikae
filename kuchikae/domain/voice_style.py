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
    if audio_emotion is None or audio_emotion.confidence < 0.5:
        return text_style
    speed = text_style.speed
    emphasis = text_style.emphasis
    if audio_emotion.energy == "high" and text_style.mood == VoiceMood.URGENT:
        speed = VoiceSpeed.FAST
        emphasis = VoiceEmphasis.HIGH
    elif audio_emotion.energy == "low" and text_style.mood in {VoiceMood.CALM, VoiceMood.APOLOGETIC}:
        speed = VoiceSpeed.SLOW
    return VoiceStyle(
        mood=text_style.mood,
        speed=speed,
        clarity=text_style.clarity,
        emphasis=emphasis,
        confidence=text_style.confidence,
        source=f"{text_style.source}+{audio_emotion.source}",
    )


VOICE_STYLE_PRESETS = {
    "natural": "自然で聞き取りやすく、元話者の声質に近い雰囲気で読んでください。文章内容は変えないでください。",
    "calm": "落ち着いた声で、自然な速さで、聞き取りやすく読んでください。文章内容は変えないでください。",
    "bright": "明るく、自然な抑揚で、聞き取りやすく読んでください。文章内容は変えないでください。",
    "slow_clear": "少しゆっくり、明瞭に、聞き取りやすく読んでください。文章内容は変えないでください。",
}
