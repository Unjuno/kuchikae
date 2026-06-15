"""Transformers-based Whisper ASR backend for Japanese speech."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import soundfile as sf

from kuchikae.domain.audio import linear_resample, torch_module
from kuchikae.domain.stt import STTBackend

logger = logging.getLogger(__name__)

_DEFAULT_WHISPER_MODEL_ID = "kotoba-tech/kotoba-whisper-v2.1"


class TransformersWhisperJapaneseASRBackend(STTBackend):
    """Japanese ASR using Hugging Face Transformers Whisper models."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        torch_dtype: str | None = None,
    ) -> None:
        self._model_id = model_id or os.environ.get("TRANSFORMERS_WHISPER_STT_MODEL_ID", _DEFAULT_WHISPER_MODEL_ID)
        self._device = device or os.environ.get("TRANSFORMERS_WHISPER_STT_DEVICE", "cpu")
        self._torch_dtype = torch_dtype or os.environ.get("TRANSFORMERS_WHISPER_STT_TORCH_DTYPE", "float32")
        self._processor = None
        self._model = None
        self._resolved_model_id: str | None = None
        self._resolved_device: str | None = None
        self._resolved_torch_dtype: str | None = None
        try:
            import transformers  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "TransformersWhisperJapaneseASRBackend requires the `transformers` package. "
                "Install with `uv sync --extra real`."
            ) from e

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._processor, self._model

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        torch = torch_module()
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self._torch_dtype, torch.float32)

        logger.info(
            "loading transformers whisper STT model '%s' (device=%s dtype=%s)...",
            self._model_id,
            self._device,
            self._torch_dtype,
        )
        t0 = time.time()
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(self._model_id, torch_dtype=dtype)
        if self._device != "cpu":
            self._model = self._model.to(self._device)
        self._model.eval()
        self._resolved_model_id = self._model_id
        self._resolved_device = self._device
        self._resolved_torch_dtype = self._torch_dtype
        logger.info("transformers whisper STT model loaded in %.2fs", time.time() - t0)
        return self._processor, self._model

    def _prepare_inputs(self, audio_path: str) -> dict[str, Any]:
        processor, _ = self._load_model()
        samples, sample_rate = sf.read(audio_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        samples = linear_resample(samples.astype(np.float32, copy=False), sample_rate, 16000)
        return processor(samples, sampling_rate=16000, return_tensors="pt")

    def transcribe(self, audio_path: str) -> str:
        processor, model = self._load_model()
        torch = torch_module()
        inputs = self._prepare_inputs(audio_path)

        t1 = time.time()
        with torch.no_grad():
            input_features = inputs["input_features"]
            if self._device != "cpu":
                input_features = input_features.to(self._device)
            generated_ids = model.generate(
                input_features,
                language="ja",
                task="transcribe",
            )
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info("transformers whisper STT transcribe: %.2fs", time.time() - t1)
        return text.strip()
