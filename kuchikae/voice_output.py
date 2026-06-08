"""VoiceOutputBackend and DummyVoiceOutputBackend."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import soundfile as sf

OUTPUT_DIR = "outputs"


class VoiceOutputBackend:
    """Abstract base for voice output backends."""

    def synthesize(  # pragma: no cover
        self,
        text: str,
        voice_context: Any,
        prompt: Any,
    ) -> str:
        raise NotImplementedError


class DummyVoiceOutputBackend(VoiceOutputBackend):
    """Creates a valid short silent WAV file at outputs/dummy.wav."""

    def synthesize(  # type: ignore[override]
        self,
        text: str,
        voice_context: Any,
        prompt: Any,
    ) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "dummy.wav")

        duration_seconds = 1.0
        sample_rate = 44_100
        samples = np.zeros(int(duration_seconds * sample_rate), dtype=np.float32)
        sf.write(output_path, samples, sample_rate)
        return output_path
