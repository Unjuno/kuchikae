"""Core data types for Kuchikae."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProsodyProfile:
    """Captured prosodic characteristics of a voice."""

    speech_rate_chars_per_sec: float | None = None
    mean_pitch_hz: float | None = None
    rms_energy: float | None = None


@dataclass
class VoiceContext:
    """Voice identity + captured prosody from a reference audio."""

    voice_id: str
    ready: bool = False
    speaker_embedding: object | None = None
    prosody_profile: ProsodyProfile | None = None
    reference_audio_path: str = ""


@dataclass
class TextTransformPrompt:
    """Free-form text instruction for the transformation."""

    prompt_text: str

    @classmethod
    def from_file(cls, path: str) -> "TextTransformPrompt":
        with open(path, encoding="utf-8") as f:
            return cls(prompt_text=f.read().strip())


@dataclass
class VoiceOutputPrompt:
    """Free-form text instruction for the output voice."""

    prompt_text: str

    @classmethod
    def from_file(cls, path: str) -> "VoiceOutputPrompt":
        with open(path, encoding="utf-8") as f:
            return cls(prompt_text=f.read().strip())


@dataclass
class LatencyReport:
    """Timing for each pipeline stage."""

    stt_seconds: float = 0.0
    text_transform_seconds: float = 0.0
    voice_output_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass
class PipelineResult:
    """Final output of the Kuchikae pipeline."""

    source_text: str
    transformed_text: str
    output_audio_path: str
    text_transform_prompt: str
    voice_output_prompt: str
    voice_ready: bool = False
    latency: LatencyReport = field(default_factory=LatencyReport)
