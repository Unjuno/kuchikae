"""Counting test doubles for performance testing."""

from __future__ import annotations

from kuchikae.domain.stt import STTBackend, DummySTTBackend
from kuchikae.domain.text_transform import TextTransformBackend, DummyTextTransformBackend
from kuchikae.domain.types import TextTransformPrompt, VoiceContext, VoiceOutputPrompt
from kuchikae.domain.voice_output import VoiceOutputBackend, DummyVoiceOutputBackend
from kuchikae.domain.audio_cache import DummyVoiceContextExtractor


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
        self.last_text: str | None = None
        self.last_voice_context: VoiceContext | None = None
        self.last_prompt: VoiceOutputPrompt | None = None

    def synthesize(
        self,
        text: str,
        voice_context: VoiceContext,
        prompt: VoiceOutputPrompt | None = None,
    ) -> str:
        self.call_count += 1
        self.last_text = text
        self.last_voice_context = voice_context
        self.last_prompt = prompt
        return self.inner.synthesize(text, voice_context, prompt)

    def reset(self) -> None:
        self.call_count = 0
        self.last_text = None
        self.last_voice_context = None
        self.last_prompt = None
