"""Voice output backend interfaces."""

from __future__ import annotations

import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)
OUTPUT_DIR = "outputs"


class VoiceOutputBackend(ABC):
    """Abstract base for voice-conditioned output backends."""

    @abstractmethod
    def synthesize(self, text: str, audio_path: str) -> str:
        """Synthesize output audio and return its file path."""
        raise NotImplementedError


class DummyVoiceOutputBackend(VoiceOutputBackend):
    """Create a valid short silent WAV file for v0.1 scaffold tests."""

    def synthesize(self, text: str, audio_path: str) -> str:
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
    - OPENVOICE_READY=1 env set.

    The backend adds the OpenVoice repo path to sys.path on first use, so it does not
    pollute the global import namespace.
    """

    def __init__(self, openvoice_path: str | None = None) -> None:
        self._openvoice_path = openvoice_path or os.environ.get(
            "KUCHIKAE_OPENVOICE_PATH", "/Users/taka/repos/OpenVoice"
        )
        self._base_tts: Any = None
        self._converter: Any = None

    def _log(self, msg: str) -> None:
        logger.info("[OpenVoice] %s", msg)

    def _ensure_models_loaded(self) -> tuple[Any, Any]:
        if self._base_tts is not None and self._converter is not None:
            return self._base_tts, self._converter

        if os.path.isdir(self._openvoice_path) and self._openvoice_path not in sys.path:
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
        import torch
        import torchaudio
        import openvoice.se_extractor as se_extractor_module

        waveforms, sr = torchaudio.load(audio_path)
        if waveforms.shape[0] > 1:
            waveforms = torch.mean(waveforms, dim=0, keepdim=True)

        chunk_samples = int(2.0 * sr)
        se_list = []
        num_chunks = max(1, waveforms.shape[1] // chunk_samples)

        for i in range(num_chunks):
            se_list.append(se_extractor_module.extract_se(
                audio_path, device="cpu", fsave_se=False, save_path=OUTPUT_DIR,
            ))

        if len(se_list) == 1:
            return se_list[0]
        return torch.mean(torch.stack(se_list), dim=0)

    def synthesize(self, text: str, audio_path: str) -> str:
        """Synthesize output audio using OpenVoice.

        Pipeline:
          1. Lazy-load models (BaseSpeakerTTS + ToneColorConverter).
          2. Multi-frame SE extraction from input audio for tone color.
          3. Synthesize with BaseSpeakerTTS preserving the extracted voice.
        """
        base_tts, converter = self._ensure_models_loaded()

        target_se = self._extract_multi_frame_se(audio_path)
        self._log(f"Multi-frame SE extracted ({text[:40]}...)")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "openvoice_output.wav")

        base_tts.tts(
            text=text,
            output_path=output_path,
            speaker="en_default",
            language="English",
            style_se=target_se,
            device="cpu",
        )

        self._log(f"Synthesized: {text[:40]}... -> {output_path}")
        return output_path


class IrodoriTTSVoiceOutputBackend(VoiceOutputBackend):
    """Japanese voice-conditioned output using Irodori-TTS.

    Uses Aratako/Irodori-TTS-500M-v3 (or v3 VoiceDesign) for zero-shot
    voice cloning from reference audio. Model weights auto-download from
    HuggingFace on first use.
    """

    def __init__(
        self,
        hf_checkpoint: str = "Aratako/Irodori-TTS-500M-v3",
        codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim",
    ) -> None:
        self._hf_checkpoint = hf_checkpoint
        self._codec_repo = codec_repo
        self._runtime: Any = None

    def _log(self, msg: str) -> None:
        logger.info("[IrodoriTTS] %s", msg)

    def _ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime

        from irodori_tts.inference_runtime import (
            InferenceRuntime,
            RuntimeKey,
            default_runtime_device,
        )

        device = default_runtime_device()  # "mps" on Apple Silicon
        self._log(f"Loading model {self._hf_checkpoint} on {device}...")

        self._runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=self._hf_checkpoint,
                model_device=device,
                codec_repo=self._codec_repo,
                codec_device=device,
                codec_deterministic_encode=True,
                codec_deterministic_decode=True,
            )
        )
        self._log("Model loaded")
        return self._runtime

    def synthesize(self, text: str, audio_path: str) -> str:
        """Synthesize output audio using Irodori-TTS voice cloning."""
        import torch  # noqa: F811

        runtime = self._ensure_runtime()
        from irodori_tts.inference_runtime import SamplingRequest, save_wav

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "irodori_output.wav")

        result = runtime.synthesize(
            SamplingRequest(
                text=text,
                ref_wav=audio_path,
                num_steps=40,
                cfg_scale_text=3.0,
                cfg_scale_speaker=5.0,
                trim_tail=True,
            )
        )

        saved = save_wav(output_path, result.audio, result.sample_rate)
        self._log(f"Synthesized: {text[:40]}... -> {saved}")
        return str(saved)
