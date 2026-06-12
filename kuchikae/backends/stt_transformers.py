"""Transformers-based Japanese ASR backends."""

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

_DEFAULT_HUBERT_MODEL_ID = "TKU410410103/hubert-large-japanese-asr"


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


class TransformersJapaneseASRBackend(STTBackend):
    """Japanese ASR using Hugging Face Transformers.

    The default model is TKU410410103/hubert-large-japanese-asr, which outputs
    hiragana and was trained on ReazonSpeech / Common Voice 11.0.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        torch_dtype: str | None = None,
    ) -> None:
        self._model_id = model_id or os.environ.get("TRANSFORMERS_STT_MODEL_ID", _DEFAULT_HUBERT_MODEL_ID)
        self._device = device or os.environ.get("TRANSFORMERS_STT_DEVICE", "cpu")
        self._torch_dtype = torch_dtype or os.environ.get("TRANSFORMERS_STT_TORCH_DTYPE", "float32")
        self._processor = None
        self._model = None
        self._resolved_model_id: str | None = None
        self._resolved_device: str | None = None
        self._resolved_torch_dtype: str | None = None
        try:
            import transformers  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "TransformersJapaneseASRBackend requires the `transformers` package. "
                "Install with `uv sync --extra real`."
            ) from e

    @staticmethod
    @lru_cache(maxsize=1)
    def _torch_module():
        import torch

        return torch

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._processor, self._model

        from transformers import AutoModelForCTC, AutoProcessor

        torch = self._torch_module()
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self._torch_dtype, torch.float32)

        logger.info(
            "loading transformers STT model '%s' (device=%s dtype=%s)...",
            self._model_id,
            self._device,
            self._torch_dtype,
        )
        t0 = time.time()
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForCTC.from_pretrained(self._model_id, torch_dtype=dtype)
        if self._device != "cpu":
            self._model = self._model.to(self._device)
        self._model.eval()
        self._resolved_model_id = self._model_id
        self._resolved_device = self._device
        self._resolved_torch_dtype = self._torch_dtype
        logger.info("transformers STT model loaded in %.2fs", time.time() - t0)
        return self._processor, self._model

    def _prepare_inputs(self, audio_path: str) -> dict[str, Any]:
        processor, _ = self._load_model()
        samples, sample_rate = sf.read(audio_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        samples = _linear_resample(samples.astype(np.float32, copy=False), sample_rate, 16000)
        return processor(samples, sampling_rate=16000, return_tensors="pt")

    def transcribe(self, audio_path: str) -> str:
        processor, model = self._load_model()
        torch = self._torch_module()
        inputs = self._prepare_inputs(audio_path)

        t1 = time.time()
        with torch.no_grad():
            input_values = inputs["input_values"]
            if self._device != "cpu":
                input_values = input_values.to(self._device)
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            text = processor.batch_decode(predicted_ids)[0]
        logger.info("transformers STT transcribe: %.2fs", time.time() - t1)
        return text.strip()

