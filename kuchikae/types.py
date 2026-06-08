"""Core data types for Kuchikae."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProsodyProfile:
    """Captured prosodic characteristics of a voice."""

    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    speed: float = 0.0
    volume: float = 0.0


@dataclass
class VoiceContext:
    """Voice identity + captured prosody from a reference audio."""

    voice_id: str
    prosody: ProsodyProfile = field(default_factory=ProsodyProfile)
    ready: bool = False
    reference_path: str | None = None


@dataclass
class TextTransformPrompt:
    """Free-form text instruction for the transformation."""

    prompt_text: str

    @classmethod
    def from_file(cls, path: str) -> TextTransformPrompt:
        with open(path, encoding="utf-8") as f:
            return cls(prompt_text=f.read().strip())


@dataclass
class VoiceOutputPrompt:
    """Free-form text instruction for the output voice."""

    prompt_text: str

    @classmethod
    def from_file(cls, path: str) -> VoiceOutputPrompt:
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
    voice_context: VoiceContext
    latency: LatencyReport
