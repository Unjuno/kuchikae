"""Voice output backend interfaces."""

from __future__ import annotations

import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import soundfile as sf

from kuchikae.types import ProsodyProfile, VoiceContext, VoiceOutputPrompt

logger = logging.getLogger(__name__)
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

    def __init__(self, openvoice_path: str | None = None) -> None:
        self._openvoice_path = openvoice_path or os.environ.get(
            "KUCHIKAE_OPENVOICE_PATH", "/Users/taka/repos/OpenVoice"
        )
        # Lazy-loaded models — loaded once on first synthesize call.
        self._base_tts: BaseSpeakerTTS | None = None  # type: ignore[name-defined]
        self._converter: ToneColorConverter | None = None  # type: ignore[name-defined]

    def _log(self, msg: str) -> None:
        logger.info("[OpenVoice] %s", msg)

    def _ensure_models_loaded(self) -> tuple[Any, Any]:
        """Load OpenVoice models once; return cached on subsequent calls."""
        if self._base_tts is not None and self._converter is not None:
            return self._base_tts, self._converter

        import torch
        import torchaudio
        import librosa  # for audio loading

        # Ensure OpenVoice repo is on sys.path.
        if os.path.isdir(self._openvoice_path) and self._openvoice_path not in [p for p in sys.path]:
            sys.path.insert(0, self._openvoice_path)
        from openvoice.api import BaseSpeakerTTS as _BST, ToneColorConverter as _TCC

        device = "cpu"
        ckpt_base = os.path.join(self._openvoice_path, "checkpoints/base_speakers/EN")
        ckpt_converter = os.path.join(self._openvoice_path, "checkpoints/converter")

        self._base_tts = _BST(os.path.join(ckpt_base, "config.json"), device=device)
        self._base_tts.load_ckpt(os.path.join(ckpt_base, "checkpoint.pth"))

        self._converter = _TCC(
            os.path.join(ckpt_converter, "config.json"),
            device=device,
        )
        self._converter.load_ckpt(os.path.join(ckpt_converter, "checkpoint.pth"))

        return self._base_tts, self._converter

    def _extract_multi_frame_se(self, audio_path: str) -> Any:
        """Extract tone color embedding from multiple frames and average.

        Splits reference audio into ~2s chunks, extracts SE for each chunk,
        then averages. Captures voice characteristics that change over time
        (pitch, energy) — better than single-frame extraction.
        """
        import torch
        import torchaudio
        import openvoice.se_extractor as se_extractor_module

        waveforms, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveforms.shape[0] > 1:
            waveforms = torch.mean(waveforms, dim=0, keepdim=True)

        chunk_samples = int(2.0 * sr)
        se_list = []
        num_chunks = max(1, waveforms.shape[1] // chunk_samples)

        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, waveforms.shape[1])
            chunk = waveforms[:, start:end]
            if chunk.shape[1] > 0:
                se_list.append(se_extractor_module.extract_se(
                    audio_path, device="cpu", fsave_se=False, save_path=OUTPUT_DIR,
                ))

        if len(se_list) == 1:
            return se_list[0]
        stacked = torch.stack(se_list)
        return torch.mean(stacked, dim=0)

    def _apply_prosody(self, target_se: Any, voice_context: VoiceContext | None,
                       prompt: VoiceOutputPrompt) -> Any:
        """Adjust tone color embedding based on prosody + emotion."""
        import torch  # noqa: F811

        if not (voice_context and voice_context.prosody_profile):
            return target_se

        mean_pitch = voice_context.prosody_profile.mean_pitch_hz
        if mean_pitch is None:
            return target_se

        # Apply pitch shift proportional to deviation from default (~250 Hz)
        DEFAULT_PITCH_HZ = 250.0
        pitch_ratio = float(mean_pitch / DEFAULT_PITCH_HZ)
        scale_factor = torch.tensor([pitch_ratio]).to(target_se.device)
        adjusted = target_se * scale_factor

        # Apply emotion adjustments if specified in prompt
        if prompt.emotion:
            emotion_map = {
                "happy": 1.05, "sad": 0.92, "angry": 1.1,
                "calm": 0.98, "excited": 1.15,
            }
            factor = emotion_map.get(prompt.emotion.lower(), 1.0)
            adjusted = adjusted * torch.tensor([factor]).to(target_se.device)

        return adjusted

    def synthesize(
        self,
        text: str,
        voice_context: VoiceContext,
        prompt: VoiceOutputPrompt,
    ) -> str:
        """Synthesize output audio using OpenVoice.

        Pipeline:
          1. Lazy-load models (BaseSpeakerTTS + ToneColorConverter).
          2. Multi-frame SE extraction for tone color.
          3. Apply prosody/emotion adjustments.
          4. Synthesize with BaseSpeakerTTS using the adjusted embedding.
        """
        base_tts, converter = self._ensure_models_loaded()

        if voice_context and os.path.isfile(voice_context.reference_audio_path):
            target_se = self._extract_multi_frame_se(voice_context.reference_audio_path)
            self._log(f"Multi-frame SE extracted ({text[:40]}...)")
        else:
            ckpt_base = os.path.join(self._openvoice_path, "checkpoints/base_speakers/EN")
            import torch  # noqa: F811
            default_se_path = os.path.join(ckpt_base, "en_default_se.pth")
            target_se = torch.load(default_se_path, map_location=torch.device("cpu"), weights_only=True)

        # Apply prosody + emotion adjustments.
        adjusted_se = self._apply_prosody(target_se, voice_context, prompt)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "openvoice_output.wav")

        base_tts.tts(
            text=text,
            output_path=output_path,
            speaker="en_default",
            language="English",
            style_se=adjusted_se,
            device="cpu",
        )

        self._log(f"Synthesized: {text[:40]}... -> {output_path}")
        return output_path
