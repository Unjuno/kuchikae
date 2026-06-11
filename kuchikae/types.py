"""Core data types for Kuchikae."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from typing import NamedTuple


@dataclass
class TextTransformPrompt:
    instruction: str

    @classmethod
    def from_file(cls, path: str | None = None) -> "TextTransformPrompt":
        if path is not None:
            with open(path, encoding="utf-8") as f:
                return cls(instruction=f.read().strip())
        text = files("kuchikae.prompts").joinpath("text_transform_default.txt").read_text(encoding="utf-8")
        return cls(instruction=text.strip())


@dataclass
class VoiceOutputPrompt:
    instruction: str

    @classmethod
    def from_file(cls, path: str | None = None) -> "VoiceOutputPrompt":
        if path is not None:
            with open(path, encoding="utf-8") as f:
                return cls(instruction=f.read().strip())
        text = files("kuchikae.prompts").joinpath("voice_output_default.txt").read_text(encoding="utf-8")
        return cls(instruction=text.strip())


@dataclass
class PipelineResult:
    output_audio_path: str
    source_text: str = ""
    transformed_text: str = ""


class AudioCacheKey(NamedTuple):
    path: str
    size: int
    mtime: float

    @classmethod
    def from_file(cls, path: str) -> "AudioCacheKey":
        st = os.stat(path)
        return cls(path=os.path.abspath(path), size=st.st_size, mtime=st.st_mtime)


@dataclass
class AudioSegment:
    start_sec: float
    end_sec: float
    samples: object
    sample_rate: int


@dataclass
class VoiceContext:
    reference_audio_path: str
    ready: bool
    speaker_embedding: object | None = None
    prosody_profile: object | None = None
