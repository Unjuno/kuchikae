"""NeMo-based Japanese ASR backends."""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any

import numpy as np
import soundfile as sf

from kuchikae.domain.stt import STTBackend

logger = logging.getLogger(__name__)

_DEFAULT_NEMO_MODEL_ID = "reazon-research/reazonspeech-nemo-v2"


def _linear_resample(samples: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.shape[0] / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32, copy=False)


class ReazonSpeechNemoASRBackend(STTBackend):
    """Japanese ASR using NVIDIA NeMo checkpoints.

    The default model is reazon-research/reazonspeech-nemo-v2, which is a
    long-form Japanese ASR model trained on ReazonSpeech v2.0.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_id = model_id or os.environ.get("NEMO_STT_MODEL_ID", _DEFAULT_NEMO_MODEL_ID)
        self._device = device or os.environ.get("NEMO_STT_DEVICE", "cpu")
        self._model = None
        self._resolved_model_id: str | None = None
        self._resolved_device: str | None = None
        try:
            import nemo.collections.asr as nemo_asr  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "ReazonSpeechNemoASRBackend requires the `nemo_toolkit[asr]` package. "
                "Install it in an optional environment before using this backend."
            ) from e

    @staticmethod
    @lru_cache(maxsize=1)
    def _torch_module():
        import torch

        return torch

    def _load_model(self):
        if self._model is not None:
            return self._model

        import nemo.collections.asr as nemo_asr

        logger.info("loading NeMo STT model '%s' (device=%s)...", self._model_id, self._device)
        t0 = time.time()
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self._model_id)
        if self._device != "cpu" and hasattr(self._model, "to"):
            self._model = self._model.to(self._device)
        self._resolved_model_id = self._model_id
        self._resolved_device = self._device
        logger.info("NeMo STT model loaded in %.2fs", time.time() - t0)
        return self._model

    def transcribe(self, audio_path: str) -> str:
        model = self._load_model()
        t1 = time.time()
        outputs = model.transcribe([audio_path])
        logger.info("NeMo transcribe: %.2fs", time.time() - t1)
        if not outputs:
            return ""
        first = outputs[0]
        text = getattr(first, "text", first)
        return str(text).strip()

    def transcribe_stream(self, audio_path: str):
        yield self.transcribe(audio_path)

