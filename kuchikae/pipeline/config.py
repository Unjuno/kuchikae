"""Pipeline configuration data classes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class STTConfig:
    """STT backend configuration."""
    backend: str = "faster_whisper"
    preset: str = "balanced"
    model_size: str | None = None
    model_id: str | None = None
    device: str | None = None
    compute_type: str | None = None
    language: str | None = None
    beam_size: int | None = None
    vad_filter: bool | None = None
    temperature: float | None = None
    condition_on_previous_text: bool | None = None
    segmented: bool = False
    streaming: bool = False
    timeout_sec: float = 120.0

    @classmethod
    def from_preset(cls, preset_name: str) -> STTConfig:
        """Create config from a preset name."""
        from kuchikae.domain.stt import resolve_stt_preset
        preset = resolve_stt_preset(preset_name)
        return cls(
            preset=preset_name,
            model_size=preset.model_size,
            device=preset.device,
            compute_type=preset.compute_type,
            language=preset.language,
            beam_size=preset.beam_size,
            vad_filter=preset.vad_filter,
            temperature=preset.temperature,
            condition_on_previous_text=preset.condition_on_previous_text,
        )


@dataclass
class TextTransformConfig:
    """Text transform backend configuration."""
    backend: str = "prompted_rule"
    model: str | None = None
    strict: bool = True
    timeout_sec: float = 60.0

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = os.environ.get("KUCHIKAE_TEXT_MODEL", "gemma3:1b-it-qat")


@dataclass
class VoiceOutputConfig:
    """Voice output backend configuration."""
    backend: str = "irodori"
    timeout_sec: float = 120.0


@dataclass
class AudioEmotionConfig:
    """Audio emotion detection configuration."""
    detector: str = "auto"
    model_id: str | None = None
    strict: bool = False
    timeout_sec: float = 5.0


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    disable_processing_cache: bool = False


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    stt: STTConfig = field(default_factory=STTConfig)
    text_transform: TextTransformConfig = field(default_factory=TextTransformConfig)
    voice_output: VoiceOutputConfig = field(default_factory=VoiceOutputConfig)
    audio_emotion: AudioEmotionConfig = field(default_factory=AudioEmotionConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    allow_dummy_backends: bool = False

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> PipelineConfig:
        """Create PipelineConfig from a legacy config dict."""
        stt_preset = config.get("stt_preset", "balanced")

        stt = STTConfig(
            backend=config.get("stt_backend", "faster_whisper"),
            preset=stt_preset,
            model_size=config.get("stt_model_size"),
            model_id=config.get("stt_model_id"),
            device=config.get("stt_device"),
            compute_type=config.get("stt_compute_type"),
            language=config.get("stt_language"),
            beam_size=config.get("stt_beam_size"),
            vad_filter=config.get("stt_vad_filter"),
            temperature=config.get("stt_temperature"),
            condition_on_previous_text=config.get("stt_condition_on_previous_text"),
            segmented=config.get("segmented_stt", False),
            streaming=config.get("streaming_stt", False),
            timeout_sec=float(config.get("stt_timeout_sec", 120.0)),
        )

        text_transform = TextTransformConfig(
            backend=config.get("text_transform_backend", "prompted_rule"),
            model=config.get("text_transform_model"),
            strict=not config.get("allow_dummy_backends", False),
            timeout_sec=float(config.get("text_transform_timeout_sec", 60.0)),
        )

        voice_output = VoiceOutputConfig(
            backend=config.get("voice_output_backend", "irodori"),
            timeout_sec=float(config.get("tts_timeout_sec", 120.0)),
        )

        audio_emotion = AudioEmotionConfig(
            detector=config.get("audio_emotion_detector", "auto"),
            model_id=config.get("audio_emotion_model_id"),
            strict=config.get("audio_emotion_strict", False),
            timeout_sec=float(config.get("voice_style_timeout_sec", 5.0)),
        )

        cache = CacheConfig(
            enabled=not config.get("disable_processing_cache", False),
            disable_processing_cache=bool(
                config.get("disable_processing_cache", False)
                or os.environ.get("KUCHIKAE_DISABLE_PROCESSING_CACHE", "").lower() in ("1", "true", "yes")
            ),
        )

        return cls(
            stt=stt,
            text_transform=text_transform,
            voice_output=voice_output,
            audio_emotion=audio_emotion,
            cache=cache,
            allow_dummy_backends=config.get("allow_dummy_backends", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy config dict for backward compatibility."""
        return {
            "stt_backend": self.stt.backend,
            "stt_preset": self.stt.preset,
            "stt_model_size": self.stt.model_size,
            "stt_model_id": self.stt.model_id,
            "stt_device": self.stt.device,
            "stt_compute_type": self.stt.compute_type,
            "stt_language": self.stt.language,
            "stt_beam_size": self.stt.beam_size,
            "stt_vad_filter": self.stt.vad_filter,
            "stt_temperature": self.stt.temperature,
            "stt_condition_on_previous_text": self.stt.condition_on_previous_text,
            "segmented_stt": self.stt.segmented,
            "streaming_stt": self.stt.streaming,
            "stt_timeout_sec": self.stt.timeout_sec,
            "text_transform_backend": self.text_transform.backend,
            "text_transform_model": self.text_transform.model,
            "text_transform_timeout_sec": self.text_transform.timeout_sec,
            "voice_output_backend": self.voice_output.backend,
            "tts_timeout_sec": self.voice_output.timeout_sec,
            "audio_emotion_detector": self.audio_emotion.detector,
            "audio_emotion_model_id": self.audio_emotion.model_id,
            "audio_emotion_strict": self.audio_emotion.strict,
            "voice_style_timeout_sec": self.audio_emotion.timeout_sec,
            "disable_processing_cache": self.cache.disable_processing_cache,
            "allow_dummy_backends": self.allow_dummy_backends,
        }
