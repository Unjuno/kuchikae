"""Kuchikae domain package.

This package exposes the lightweight domain model only. Heavy runtime backends
live under :mod:`kuchikae.backends`.
"""

from kuchikae.domain.audio import AudioSegmenter, FixedWindowSegmenter, TranscriptJoiner
from kuchikae.domain.audio_cache import AudioCache, VoiceContextExtractor
from kuchikae.domain.audio_key import AudioKey, AudioKeyFromCacheKey, AudioKeyFromPath
from kuchikae.domain.audio_stream import AudioChunk, AudioChunker, AudioStreamBuffer, EnergyVAD
from kuchikae.domain.metrics import LatencyLogger, StreamingMetricsRecorder
from kuchikae.domain.processing_cache import ProcessingCache
from kuchikae.domain.stt import (
    DummySTTBackend,
    DummyStreamingSTTBackend,
    SegmentedSTTBackend,
    STTBackend,
    StreamingSTTBackend,
)
from kuchikae.domain.text_transform import (
    DummyIncrementalTextTransformBackend,
    DummyTextTransformBackend,
    IncrementalTextTransformBackend,
    PromptedRuleTextTransformBackend,
    RuleTextTransformBackend,
    TemplateTextTransformBackend,
    TextTransformBackend,
)
from kuchikae.domain.timing import PerfTimer, Timer, now_ms
from kuchikae.domain.types import (
    AudioCacheKey,
    AudioSegment,
    PipelineResult,
    STTCommit,
    STTFinal,
    STTPartial,
    StreamChunk,
    StreamingAudioSegment,
    StreamingLatencyReport,
    TextTransformPrompt,
    TransformState,
    TransformUpdate,
    VoiceContext,
    VoiceOutputPrompt,
)
from kuchikae.domain.voice_output import (
    AudioSegmentQueue,
    DummyStreamingVoiceOutputBackend,
    DummyVoiceOutputBackend,
    StreamingVoiceOutputBackend,
    VoiceOutputBackend,
    segment_clauses,
    segment_sentences,
)

__all__ = [
    "AudioCache",
    "AudioCacheKey",
    "AudioChunk",
    "AudioChunker",
    "AudioKey",
    "AudioKeyFromCacheKey",
    "AudioKeyFromPath",
    "AudioSegment",
    "AudioSegmenter",
    "AudioSegmentQueue",
    "AudioStreamBuffer",
    "DummyIncrementalTextTransformBackend",
    "DummyStreamingSTTBackend",
    "DummyStreamingVoiceOutputBackend",
    "DummySTTBackend",
    "DummyTextTransformBackend",
    "DummyVoiceOutputBackend",
    "EnergyVAD",
    "FixedWindowSegmenter",
    "IncrementalTextTransformBackend",
    "LatencyLogger",
    "PerfTimer",
    "PipelineResult",
    "ProcessingCache",
    "PromptedRuleTextTransformBackend",
    "RuleTextTransformBackend",
    "SegmentedSTTBackend",
    "STTBackend",
    "STTCommit",
    "STTFinal",
    "STTPartial",
    "StreamChunk",
    "StreamingAudioSegment",
    "StreamingLatencyReport",
    "StreamingMetricsRecorder",
    "StreamingSTTBackend",
    "StreamingVoiceOutputBackend",
    "TemplateTextTransformBackend",
    "TextTransformBackend",
    "TextTransformPrompt",
    "Timer",
    "TranscriptJoiner",
    "TransformState",
    "TransformUpdate",
    "VoiceContext",
    "VoiceContextExtractor",
    "VoiceOutputBackend",
    "VoiceOutputPrompt",
    "now_ms",
    "segment_clauses",
    "segment_sentences",
]
