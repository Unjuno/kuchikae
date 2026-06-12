"""Voice prompt resolution logic extracted from pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict
from typing import Callable

from kuchikae.domain.audio_emotion import (
    AudioEmotion,
    AudioEmotionDetector,
    DummyAudioEmotionDetector,
    DisabledAudioEmotionDetector,
)
from kuchikae.domain.events import DiagnosticEvent, EventLevel
from kuchikae.domain.voice_style import (
    VoiceStyle,
    VoiceStyleDetector,
    RuleVoiceStyleDetector,
    fuse_voice_styles,
    voice_style_to_prompt,
)
from kuchikae.domain.types import VoiceOutputPrompt
from kuchikae.logging import setup_logger

logger = setup_logger("kuchikae.voice_prompt_resolver")


class VoicePromptResolver:
    """Resolves voice output prompts from source text, transformed text, and audio emotion.

    Responsibilities:
    - Start/collect async audio emotion detection
    - Detect voice style from source and transformed text
    - Fuse voice styles with audio emotion
    - Generate voice output prompt
    - Emit diagnostics events
    """

    def __init__(
        self,
        voice_style_detector: VoiceStyleDetector | None = None,
        audio_emotion_detector: AudioEmotionDetector | None = None,
        timeout_sec: float = 0.05,
        emit: Callable[[str, str, str, EventLevel, str | None, str | None, float | None, dict | None], None] | None = None,
    ) -> None:
        self.voice_style_detector = voice_style_detector or RuleVoiceStyleDetector()
        self.audio_emotion_detector = audio_emotion_detector or DummyAudioEmotionDetector()
        self.timeout_sec = timeout_sec
        self._emit = emit or self._default_emit
        self._last_voice_style = "auto"
        self._last_audio_emotion = "disabled"

    def _default_emit(
        self,
        name: str,
        message: str,
        stage: str,
        level: EventLevel = EventLevel.INFO,
        backend: str | None = None,
        cache: str | None = None,
        elapsed_sec: float | None = None,
        data: dict | None = None,
    ) -> None:
        logger.debug("emit: %s %s", name, data)

    def start_audio_emotion(self, audio_path: str) -> tuple[object, ThreadPoolExecutor | None]:
        """Start async audio emotion detection.

        Returns (future, executor) or (None, None) if disabled.
        """
        if getattr(self.audio_emotion_detector, "disabled", False):
            return None, None
        executor = ThreadPoolExecutor(max_workers=1)
        self._emit(
            "audio_emotion.detect.start",
            "Audio emotion detection started.",
            "audio_emotion",
            EventLevel.INFO,
            type(self.audio_emotion_detector).__name__,
        )
        future = executor.submit(self.audio_emotion_detector.detect, audio_path)
        return future, executor

    def collect_audio_emotion(self, future, executor: ThreadPoolExecutor | None) -> AudioEmotion | None:
        """Collect audio emotion detection result.

        Handles timeout, errors, and emits diagnostics.
        """
        if future is None:
            return None
        audio_emotion = None
        try:
            audio_emotion = future.result(timeout=self.timeout_sec)
            detector = self.audio_emotion_detector
            if detector is not None and getattr(detector, "model_unavailable", False):
                self._emit(
                    "audio_emotion.model_unavailable",
                    "Audio emotion model was unavailable; dummy behavior was used.",
                    "audio_emotion",
                    EventLevel.WARNING,
                    type(detector).__name__,
                    None,
                    None,
                    {
                        "model_id": getattr(detector, "model_id", None),
                        "error_type": getattr(detector, "load_error_type", None),
                    },
                )
            if detector is not None and getattr(detector, "fallback_dummy", False):
                self._emit(
                    "audio_emotion.fallback_dummy",
                    "Audio emotion detection fell back to dummy behavior.",
                    "audio_emotion",
                    EventLevel.WARNING,
                    type(detector).__name__,
                    None,
                    None,
                    {
                        "model_id": getattr(detector, "model_id", None),
                        "error_type": getattr(detector, "load_error_type", None),
                    },
                )
            self._emit(
                "audio_emotion.detect.done",
                "Audio emotion detection finished.",
                "audio_emotion",
                EventLevel.INFO,
                type(self.audio_emotion_detector).__name__,
            )
        except FuturesTimeoutError:
            self._emit(
                "audio_emotion.detect.timeout",
                "Audio emotion detection timed out.",
                "audio_emotion",
                EventLevel.WARNING,
                type(self.audio_emotion_detector).__name__,
            )
        except Exception as e:
            self._emit(
                "audio_emotion.detect.failed",
                f"{type(e).__name__}: {e}",
                "audio_emotion",
                EventLevel.WARNING,
                type(self.audio_emotion_detector).__name__,
            )
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
        return audio_emotion

    def _detect_voice_style(self, text: str) -> VoiceStyle:
        """Detect voice style from text."""
        return self.voice_style_detector.detect(text)

    def build_prompt(
        self,
        source_text: str,
        transformed_text: str,
        explicit_prompt: VoiceOutputPrompt | None,
        audio_emotion: AudioEmotion | None,
    ) -> VoiceOutputPrompt:
        """Build voice output prompt from source text, transformed text, and audio emotion.

        If explicit_prompt is provided, it is returned as-is (custom prompt priority).
        Otherwise, fuse voice styles from source and transformed text with audio emotion.
        """
        if explicit_prompt is not None:
            self._last_voice_style = "custom"
            self._last_audio_emotion = getattr(audio_emotion, "source", "disabled") if audio_emotion is not None else "disabled"
            return explicit_prompt

        self._emit(
            "voice_style.detect.start",
            "Voice style detection started.",
            "voice_style",
            EventLevel.INFO,
        )

        source_style = self._detect_voice_style(source_text)
        self._emit(
            "voice_style.source.detect.done",
            "Voice style detection finished for source text.",
            "voice_style",
            EventLevel.INFO,
            None,
            None,
            None,
            {
                "mood": source_style.mood.value,
                "speed": source_style.speed.value,
                "clarity": source_style.clarity.value,
                "emphasis": source_style.emphasis.value,
                "confidence": source_style.confidence,
                "source": source_style.source,
            },
        )

        transformed_style = self._detect_voice_style(transformed_text)
        self._emit(
            "voice_style.transformed.detect.done",
            "Voice style detection finished for transformed text.",
            "voice_style",
            EventLevel.INFO,
            None,
            None,
            None,
            {
                "mood": transformed_style.mood.value,
                "speed": transformed_style.speed.value,
                "clarity": transformed_style.clarity.value,
                "emphasis": transformed_style.emphasis.value,
                "confidence": transformed_style.confidence,
                "source": transformed_style.source,
            },
        )

        final_style = fuse_voice_styles(source_style, transformed_style, audio_emotion)
        prompt = VoiceOutputPrompt(instruction=voice_style_to_prompt(final_style))
        self._last_voice_style = final_style.mood.value
        self._last_audio_emotion = getattr(audio_emotion, "source", "disabled") if audio_emotion is not None else "disabled"

        self._emit(
            "voice_style.fusion.done",
            "Voice style fusion finished.",
            "voice_style",
            EventLevel.INFO,
            None,
            None,
            None,
            {
                "source_mood": source_style.mood.value,
                "source_speed": source_style.speed.value,
                "source_emphasis": source_style.emphasis.value,
                "source_confidence": source_style.confidence,
                "transformed_mood": transformed_style.mood.value,
                "transformed_speed": transformed_style.speed.value,
                "transformed_emphasis": transformed_style.emphasis.value,
                "transformed_confidence": transformed_style.confidence,
                "audio_energy": getattr(audio_emotion, "energy", "disabled") if audio_emotion is not None else "disabled",
                "audio_arousal": getattr(audio_emotion, "arousal", 0.0) if audio_emotion is not None else 0.0,
                "audio_confidence": getattr(audio_emotion, "confidence", 0.0) if audio_emotion is not None else 0.0,
                "audio_source": getattr(audio_emotion, "source", "disabled") if audio_emotion is not None else "disabled",
                "final_mood": final_style.mood.value,
                "final_speed": final_style.speed.value,
                "final_emphasis": final_style.emphasis.value,
                "final_confidence": final_style.confidence,
                "final_source": final_style.source,
                "generated_prompt_preview": prompt.instruction[:120],
            },
        )

        return prompt

    @property
    def last_voice_style(self) -> str:
        return self._last_voice_style

    @property
    def last_audio_emotion(self) -> str:
        return self._last_audio_emotion
