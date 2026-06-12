from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AudioEmotion:
    mood: str = "neutral"
    energy: str = "medium"
    arousal: float = 0.5
    valence: float = 0.0
    confidence: float = 0.0
    source: str = "dummy"


class AudioEmotionDetector(Protocol):
    def detect(self, audio_path: str) -> AudioEmotion:
        ...


class DummyAudioEmotionDetector:
    def detect(self, audio_path: str) -> AudioEmotion:
        return AudioEmotion()

