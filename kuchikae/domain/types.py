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


@dataclass(frozen=True)
class STTPartial:
    """Partial STT result with stable/unstable separation.

    ``stable_prefix`` is the portion that will not change.
    ``unstable_suffix`` may be revised on the next chunk.

    **Never pass ``unstable_suffix`` to LLM/TTS.**  Downstream consumers
    must only use ``stable_prefix`` (or equivalently a combination of
    ``text`` minus ``unstable_suffix``) as committed text.
    """

    session_id: str = ""
    text: str = ""
    stable_prefix: str = ""
    unstable_suffix: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.text != self.stable_prefix + self.unstable_suffix:
            raise ValueError(
                f"text must equal stable_prefix + unstable_suffix, "
                f"got text={self.text!r} vs "
                f"{self.stable_prefix!r} + {self.unstable_suffix!r}"
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be None or in [0, 1], got {self.confidence}"
            )


@dataclass
class STTFinal:
    """Final STT result for a completed utterance."""

    text: str
    start_sec: float
    end_sec: float
    confidence: float | None = None


@dataclass(frozen=True)
class STTCommit:
    """Committed STT segment in a streaming session.

    Unlike STTFinal (which marks end-of-utterance), STTCommit is emitted
    for each stable segment within a session.
    """

    session_id: str = ""
    text: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0
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


@dataclass(frozen=True)
class TransformUpdate:
    """Result of an incremental text transform operation.

    ``source_committed_text`` is the STT committed text that was input.
    ``transformed_committed_text`` is the complete transformed output so far.
    ``newly_transformed_text`` is the segment newly added in this update.

    Once a portion of text has been passed to TTS it should **not** be
    revised — this dataclass captures the append-only transform contract.
    """

    session_id: str = ""
    source_committed_text: str = ""
    transformed_committed_text: str = ""
    newly_transformed_text: str = ""
    is_final: bool = False


# ---------------------------------------------------------------------------
# Streaming audio segment type (TTS output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamingAudioSegment:
    """A single audio segment produced by streaming TTS.

    Carries text, output path, and session metadata.
    ``is_final`` marks the last segment in a flush sequence.
    """

    session_id: str = ""
    segment_index: int = 0
    text: str = ""
    audio_path: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    is_final: bool = False


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
    """Wall-clock timestamp-based latency report.

    All ``*_at`` fields are ``time.perf_counter()`` timestamps in seconds.
    Computed properties return ``None`` when input timestamps are missing.
    """

    session_id: str = ""
    recording_started_at: float | None = None
    first_partial_transcript_at: float | None = None
    first_committed_transcript_at: float | None = None
    first_transformed_text_at: float | None = None
    first_audio_at: float | None = None
    recording_finished_at: float | None = None
    processing_finished_at: float | None = None

    @property
    def time_to_first_partial_transcript(self) -> float | None:
        if self.recording_started_at is not None and self.first_partial_transcript_at is not None:
            return self.first_partial_transcript_at - self.recording_started_at
        return None

    @property
    def time_to_first_committed_transcript(self) -> float | None:
        if self.recording_started_at is not None and self.first_committed_transcript_at is not None:
            return self.first_committed_transcript_at - self.recording_started_at
        return None

    @property
    def time_to_first_transformed_text(self) -> float | None:
        if self.recording_started_at is not None and self.first_transformed_text_at is not None:
            return self.first_transformed_text_at - self.recording_started_at
        return None

    @property
    def time_to_first_audio(self) -> float | None:
        if self.recording_started_at is not None and self.first_audio_at is not None:
            return self.first_audio_at - self.recording_started_at
        return None

    @property
    def recording_duration(self) -> float | None:
        if self.recording_started_at is not None and self.recording_finished_at is not None:
            return self.recording_finished_at - self.recording_started_at
        return None

    @property
    def processing_tail_latency(self) -> float | None:
        if self.recording_finished_at is not None and self.processing_finished_at is not None:
            return self.processing_finished_at - self.recording_finished_at
        return None

    @property
    def realtime_factor(self) -> float | None:
        rd = self.recording_duration
        if rd is not None and rd > 0:
            ptl = self.processing_tail_latency
            if ptl is not None:
                return (rd + ptl) / rd
        return None
