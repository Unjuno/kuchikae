"""Pipeline step abstractions."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any

from kuchikae.domain.types import VoiceContext
from kuchikae.logging import setup_logger

logger = setup_logger("kuchikae.pipeline.steps")


@dataclass
class StepResult:
    """Result of a pipeline step execution."""
    name: str
    output: Any
    latency_sec: float
    cached: bool = False
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class PipelineStep(ABC):
    """Base class for pipeline steps."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Step name for logging and diagnostics."""

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> Any:
        """Execute the step with given context."""

    def run(self, context: dict[str, Any], timeout_sec: float = 0) -> StepResult:
        """Run step with optional timeout and timing."""
        t0 = time.time()
        cached = context.get("cached", False)
        try:
            output = self.execute(context)
            latency = time.time() - t0
            return StepResult(
                name=self.name,
                output=output,
                latency_sec=latency,
                cached=cached,
            )
        except Exception as e:
            latency = time.time() - t0
            return StepResult(
                name=self.name,
                output=None,
                latency_sec=latency,
                cached=cached,
                error=e,
            )


class STTStep(PipelineStep):
    """Speech-to-text step."""

    def __init__(self, stt_backend: Any, timeout_sec: float = 120.0):
        self._backend = stt_backend
        self._timeout_sec = timeout_sec

    @property
    def name(self) -> str:
        return "stt"

    def execute(self, context: dict[str, Any]) -> str:
        audio_path = context["audio_path"]
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._backend.transcribe, audio_path)
            return future.result(timeout=self._timeout_sec)
        except FuturesTimeoutError:
            raise TimeoutError(
                f"STT timed out after {self._timeout_sec}s "
                f"(backend={type(self._backend).__name__})"
            )
        finally:
            executor.shutdown(wait=False)


class TextTransformStep(PipelineStep):
    """Text transformation step."""

    def __init__(self, text_backend: Any, timeout_sec: float = 60.0):
        self._backend = text_backend
        self._timeout_sec = timeout_sec

    @property
    def name(self) -> str:
        return "text_transform"

    def execute(self, context: dict[str, Any]) -> str:
        source_text = context["source_text"]
        prompt = context["prompt"]
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._backend.transform, source_text, prompt)
            return future.result(timeout=self._timeout_sec)
        except FuturesTimeoutError:
            raise TimeoutError(
                f"Text transform timed out after {self._timeout_sec}s "
                f"(backend={type(self._backend).__name__})"
            )
        finally:
            executor.shutdown(wait=False)


class VoiceContextStep(PipelineStep):
    """Voice context extraction step."""

    def __init__(self, extractor: Any):
        self._extractor = extractor

    @property
    def name(self) -> str:
        return "voice_context"

    def execute(self, context: dict[str, Any]) -> VoiceContext:
        audio_path = context["audio_path"]
        return self._extractor.extract(audio_path)


class VoiceOutputStep(PipelineStep):
    """Voice output synthesis step."""

    def __init__(self, voice_backend: Any, timeout_sec: float = 120.0):
        self._backend = voice_backend
        self._timeout_sec = timeout_sec

    @property
    def name(self) -> str:
        return "voice_output"

    def execute(self, context: dict[str, Any]) -> str:
        text = context["text_for_voice"]
        voice_context = context["voice_context"]
        voice_prompt = context.get("voice_prompt")
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                self._backend.synthesize, text, voice_context, voice_prompt,
            )
            return future.result(timeout=self._timeout_sec)
        except FuturesTimeoutError:
            raise TimeoutError(
                f"TTS timed out after {self._timeout_sec}s "
                f"(backend={type(self._backend).__name__})"
            )
        finally:
            executor.shutdown(wait=False)


class AudioEmotionStep(PipelineStep):
    """Audio emotion detection step."""

    def __init__(self, detector: Any, timeout_sec: float = 5.0):
        self._detector = detector
        self._timeout_sec = timeout_sec

    @property
    def name(self) -> str:
        return "audio_emotion"

    def execute(self, context: dict[str, Any]) -> Any:
        audio_path = context["audio_path"]
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._detector.detect, audio_path)
            return future.result(timeout=self._timeout_sec)
        except FuturesTimeoutError:
            logger.warning("Audio emotion detection timed out")
            return None
        except Exception as e:
            logger.warning("Audio emotion detection failed: %s", e)
            return None
        finally:
            executor.shutdown(wait=False)
