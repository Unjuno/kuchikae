"""Counting test doubles for performance testing."""

from __future__ import annotations

from typing import Any

from kuchikae.stt import STTBackend, DummySTTBackend
from kuchikae.text_transform import TextTransformBackend, DummyTextTransformBackend
from kuchikae.types import TextTransformPrompt
from kuchikae.voice_output import VoiceOutputBackend, DummyVoiceOutputBackend
from kuchikae.voice_context import VoiceContext, DummyVoiceContextExtractor


class CountingSTTBackend(STTBackend):
    """STT backend that counts calls for testing."""

    def __init__(self, inner: STTBackend | None = None) -> None:
        self.inner = inner or DummySTTBackend()
        self.call_count = 0

    def transcribe(self, audio_path: str) -> str:
        self.call_count += 1
        return self.inner.transcribe(audio_path)

    def reset(self) -> None:
        self.call_count = 0


class CountingVoiceContextExtractor(DummyVoiceContextExtractor):
    """Voice context extractor that counts calls for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def extract(self, reference_audio_path: str | None) -> VoiceContext:
        self.call_count += 1
        return super().extract(reference_audio_path)

    def reset(self) -> None:
        self.call_count = 0


class CountingTextTransformBackend(TextTransformBackend):
    """Text transform backend that counts calls for testing."""

    def __init__(self, inner: TextTransformBackend | None = None) -> None:
        self.inner = inner or DummyTextTransformBackend()
        self.call_count = 0

    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        self.call_count += 1
        return self.inner.transform(text, prompt)

    def reset(self) -> None:
        self.call_count = 0


class CountingVoiceOutputBackend(VoiceOutputBackend):
    """Voice output backend that counts calls for testing."""

    def __init__(self, inner: VoiceOutputBackend | None = None) -> None:
        self.inner = inner or DummyVoiceOutputBackend()
        self.call_count = 0

    def synthesize(self, text: str, voice_context: VoiceContext) -> str:
        self.call_count += 1
        return self.inner.synthesize(text, voice_context)

    def reset(self) -> None:
        self.call_count = 0
