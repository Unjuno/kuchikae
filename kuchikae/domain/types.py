"""Core data types for Kuchikae."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    stt_latency: float = 0.0
    text_transform_latency: float = 0.0
    voice_output_latency: float = 0.0
    total_latency: float = 0.0


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


# ---------------------------------------------------------------------------
# Streaming STT types
# ---------------------------------------------------------------------------


@dataclass
class STTPartial:
    """Partial STT result with stable/unstable separation.

    ``stable_prefix`` is the portion that will not change.
    ``unstable_suffix`` may be revised on the next chunk.
    """

    text: str
    stable_prefix: str
    unstable_suffix: str
    start_sec: float
    end_sec: float
    confidence: float | None = None


@dataclass
class STTFinal:
    """Final STT result for a completed utterance."""

    text: str
    start_sec: float
    end_sec: float
    confidence: float | None = None


# ---------------------------------------------------------------------------
# Incremental text transform types
# ---------------------------------------------------------------------------


@dataclass
class TransformState:
    """Tracks which parts of source text have been transformed.

    ``transformed_up_to`` is the character offset in the source text
    up to which transformation has been committed.
    """

    transformed_up_to: int = 0
    accumulated_output: str = ""


@dataclass
class TransformUpdate:
    """Result of an incremental text transform operation.

    ``new_output_segment`` is the newly transformed portion that should
    be appended to previously generated output.
    ``updated_state`` reflects the new transformation state.
    """

    new_output_segment: str
    updated_state: TransformState


# ---------------------------------------------------------------------------
# Streaming / metrics types
# ---------------------------------------------------------------------------


@dataclass
class StreamChunk:
    """A single chunk in a streaming pipeline session."""

    session_id: str
    chunk_index: int
    audio_start_sec: float
    audio_end_sec: float
    partial_transcript: str = ""
    committed_transcript: str = ""
    partial_transformed_text: str = ""
    committed_transformed_text: str = ""
    output_audio_path: str | None = None
    is_final: bool = False


@dataclass
class StreamingLatencyReport:
    """Latency breakdown for a streaming pipeline session.

    All times are in seconds (SI).
    ``realtime_factor`` is auto-computed from total_recording_sec
    and total_processing_sec on construction.
    """

    session_id: str
    time_to_first_partial_transcript: float = 0.0
    time_to_first_committed_transcript: float = 0.0
    time_to_first_transformed_text: float = 0.0
    time_to_first_audio: float = 0.0
    total_recording_sec: float = 0.0
    total_processing_sec: float = 0.0
    realtime_factor: float = 0.0
    timestamp: float = 0.0
    stages: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.total_recording_sec > 0:
            self.realtime_factor = self.total_processing_sec / self.total_recording_sec
