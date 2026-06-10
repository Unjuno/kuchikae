"""Core data types for Kuchikae."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextTransformPrompt:
    """Free-form text instruction for transcript transformation."""

    instruction: str

    @classmethod
    def from_file(cls, path: str) -> "TextTransformPrompt":
        with open(path, encoding="utf-8") as f:
            return cls(instruction=f.read().strip())


@dataclass
class PipelineResult:
    """Result returned by the Kuchikae pipeline."""

    output_audio_path: str
