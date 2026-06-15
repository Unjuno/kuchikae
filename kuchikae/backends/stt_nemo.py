"""NeMo-based Japanese ASR backends."""

from __future__ import annotations

import logging
import os
import time

from typing import Any, Generator

from kuchikae.domain.stt import STTBackend

logger = logging.getLogger(__name__)

_DEFAULT_NEMO_MODEL_ID = "reazon-research/reazonspeech-nemo-v2"


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
        self._model: Any = None
        self._resolved_model_id: str | None = None
        self._resolved_device: str | None = None
        try:
            import nemo.collections.asr as nemo_asr  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "ReazonSpeechNemoASRBackend requires the `nemo_toolkit[asr]` package. "
                "Install it in an optional environment before using this backend."
            ) from e

    def _load_model(self) -> Any:
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

    def transcribe_stream(self, audio_path: str) -> Generator[str, None, None]:
        yield self.transcribe(audio_path)

