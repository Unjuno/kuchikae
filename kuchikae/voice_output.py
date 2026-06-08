"""Voice output backend interfaces."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf

from kuchikae.types import VoiceContext, VoiceOutputPrompt

OUTPUT_DIR = "outputs"


class VoiceOutputBackend(ABC):
    """Abstract base for voice-conditioned output backends."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_context: VoiceContext,
        prompt: VoiceOutputPrompt,
    ) -> str:
        """Synthesize output audio and return its file path."""
        raise NotImplementedError


class DummyVoiceOutputBackend(VoiceOutputBackend):
    """Create a valid short silent WAV file for v0.1 scaffold tests."""

    def synthesize(
        self,
        text: str,
        voice_context: VoiceContext,
        prompt: VoiceOutputPrompt,
    ) -> str:
        """Write a silent WAV while preserving the required interface shape."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "dummy.wav")

        duration_seconds = 1.0
        sample_rate = 44_100
        samples = np.zeros(int(duration_seconds * sample_rate), dtype=np.float32)
        sf.write(output_path, samples, sample_rate)
        return output_path
