"""Core data types for Kuchikae."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ProsodyProfile:
    """Optional prosodic characteristics captured from reference audio."""

    speech_rate_chars_per_sec: Optional[float] = None
    mean_pitch_hz: Optional[float] = None
    rms_energy: Optional[float] = None


@dataclass
class VoiceContext:
    """Reference-audio based voice context for voice-conditioned output."""

    reference_audio_path: str
    ready: bool
    speaker_embedding: Optional[Any] = None
    prosody_profile: Optional[ProsodyProfile] = None


@dataclass
class TextTransformPrompt:
    """Free-form text instruction for transcript transformation."""

    instruction: str
    preserve_meaning: bool = True
    max_output_chars: int = 220

    @classmethod
    def from_file(cls, path: str) -> "TextTransformPrompt":
        with open(path, encoding="utf-8") as f:
            return cls(instruction=f.read().strip())


@dat