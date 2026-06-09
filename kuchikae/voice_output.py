"""Voice output backend interfaces."""

from __future__ import annotations

import os
import sys
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


class OpenVoiceOutputBackend(VoiceOutputBackend):
    """Real voice-conditioned output using OpenVoice v2.

    Requires:
    - OpenVoice repo cloned outside this repository (default: ../OpenVoice).
    - torch, torchaudio installed and accessible via sys.path.

    The backend adds the OpenVoice repo path to sys.path on first use, so it does not
    pollute the global import namespace.
    """

    _openvoice_path: str = "/Users/taka/repos/OpenVoice"

    def synthesize(
        self,
        text: str,
        voice_context: VoiceContext,
        prompt: VoiceOutputPrompt,
    ) -> str:
        """Synthesize output audio using OpenVoice.

        1. Extract tone color embedding from reference audio (or use default).
        2. Use BaseSpeakerTTS + ToneColorConverter to produce speech with that tone color.
        """
        import torch
        import torchaudio
        import librosa
        import openvoice.se_extractor as se_extractor_module

        # Ensure OpenVoice repo is on sys.path.
        if os.path.isdir(self._openvoice_path) and self._openvoice_path not in [p for p in sys.path]:
            sys.path.insert(0, self._openvoice_path)
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter

        device = "cpu"

        ckpt_base = os.path.join(self._openvoice_path, "checkpoints/base_speakers/EN")
        ckpt_converter = os.path.join(self._openvoice_path, "checkpoints/converter")

        base_tts = BaseSpeakerTTS(os.path.join(ckpt_base, "config.json"), device=device)
        base_tts.load_ckpt(os.path.join(ckpt_base, "checkpoint.pth"))

        converter = ToneColorConverter(
            os.path.join(ckpt_converter, "config.json"),
            device=device,
        )
        converter.load_ckpt(os.path.join(ckpt_converter, "checkpoint.pth"))

        # Extract tone color embedding from reference audio.
        if voice_context and os.path.isfile(voice_context.reference_audio_path):
            target_se = se_extractor_module.extract_se(
                voice_context.reference_audio_path, device=device, fsave_se=True, save_path=OUTPUT_DIR,
            )
        else:
            default_se_path = os.path.join(ckpt_base, "en_default_se.pth")
            target_se = torch.load(default_se_path, map_location=torch.device(device), weights_only=True)

        # Generate output audio.
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "openvoice_output.wav")

        base_tts.tts(
            text=text,
            output_path=output_path,
            speaker="en_default",
            language="English",
            style_se=target_se,
            device=device,
        )

        return output_path
