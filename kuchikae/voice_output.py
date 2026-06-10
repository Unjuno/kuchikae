"""Voice output backend interfaces."""

from __future__ import annotations

import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)
OUTPUT_DIR = "outputs"


class VoiceOutputBackend(ABC):

    @abstractmethod
    def synthesize(self, text: str, audio_path: str) -> str:
        raise NotImplementedError


class DummyVoiceOutputBackend(VoiceOutputBackend):

    def synthesize(self, text: str, audio_path: str) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "dummy.wav")
        duration_seconds = 1.0
        sample_rate = 44_100
        samples = np.zeros(int(duration_seconds * sample_rate), dtype=np.float32)
        sf.write(output_path, samples, sample_rate)
        return output_path


class OpenVoiceOutputBackend(VoiceOutputBackend):

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
        t0 = time.time()
        base_tts, converter = self._ensure_models_loaded()
        self._log(f"model load: {time.time()-t0:.2f}s")

        target_se = self._extract_multi_frame_se(audio_path)
        self._log(f"SE extracted ({text[:40]}...)")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "openvoice_output.wav")

        t1 = time.time()
        base_tts.tts(
            text=text,
            output_path=output_path,
            speaker="en_default",
            language="English",
            style_se=target_se,
            device="cpu",
        )
        self._log(f"TTS: {time.time()-t1:.2f}s → {output_path}")
        return output_path


class IrodoriTTSVoiceOutputBackend(VoiceOutputBackend):

    def __init__(
        self,
        hf_checkpoint: str = "Aratako/Irodori-TTS-500M-v3",
        codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim",
        num_steps: int = 10,
    ) -> None:
        self._hf_checkpoint = hf_checkpoint
        self._codec_repo = codec_repo
        self._num_steps = num_steps
        self._runtime: Any = None

    def _log(self, msg: str) -> None:
        logger.info("[IrodoriTTS] %s", msg)

    def _ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime

        from huggingface_hub import hf_hub_download
        from irodori_tts.inference_runtime import (
            InferenceRuntime,
            RuntimeKey,
            default_runtime_device,
        )

        device = default_runtime_device()
        self._log(f"downloading model {self._hf_checkpoint}...")
        t0 = time.time()
        checkpoint_path = hf_hub_download(
            repo_id=self._hf_checkpoint,
            filename="model.safetensors",
        )
        self._log(f"download: {time.time()-t0:.0f}s")
        self._log(f"loading model on {device}...")

        t1 = time.time()
        self._runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint_path,
                model_device=device,
                codec_repo=self._codec_repo,
                codec_device=device,
                codec_deterministic_encode=True,
                codec_deterministic_decode=True,
            )
        )
        self._log(f"load: {time.time()-t1:.2f}s")
        return self._runtime

    def synthesize(self, text: str, audio_path: str) -> str:
        import torch  # noqa: F811

        t0 = time.time()
        runtime = self._ensure_runtime()
        self._log(f"runtime ready: {time.time()-t0:.2f}s")

        from irodori_tts.inference_runtime import SamplingRequest, save_wav

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "irodori_output.wav")

        self._log(f"synthesizing ({self._num_steps} steps, linear)...")
        t1 = time.time()
        result = runtime.synthesize(
            SamplingRequest(
                text=text,
                ref_wav=audio_path,
                num_steps=self._num_steps,
                t_schedule_mode="linear",
                cfg_scale_text=2.0,
                cfg_scale_speaker=3.0,
                trim_tail=True,
                context_kv_cache=True,
            )
        )
        self._log(f"inference: {time.time()-t1:.2f}s")

        saved = save_wav(output_path, result.audio, result.sample_rate)
        self._log(f"saved: {os.path.getsize(output_path)/1e6:.1f}MB")
        return str(saved)
