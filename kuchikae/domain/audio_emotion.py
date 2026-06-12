from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import os
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioEmotion:
    mood: str = "neutral"
    energy: str = "medium"
    arousal: float = 0.5
    valence: float = 0.0
    confidence: float = 0.0
    source: str = "dummy"


class AudioEmotionDetector(Protocol):
    def detect(self, audio_path: str) -> AudioEmotion:
        ...


class DummyAudioEmotionDetector:
    def detect(self, audio_path: str) -> AudioEmotion:
        return AudioEmotion()


class DisabledAudioEmotionDetector:
    disabled = True

    def detect(self, audio_path: str) -> AudioEmotion:
        return AudioEmotion(source="disabled")


class TransformersAudioEmotionDetector:
    """Lazy-loaded audio emotion detector using Hugging Face Transformers.

    The model id is configurable via KUCHIKAE_AUDIO_EMOTION_MODEL.
    If dependencies or weights are unavailable, the detector falls back to DummyAudioEmotionDetector
    behavior unless strict=True.
    """

    def __init__(self, model_id: str | None = None, strict: bool = False) -> None:
        self.model_id = model_id or os.environ.get(
            "KUCHIKAE_AUDIO_EMOTION_MODEL",
            "superb/wav2vec2-base-superb-er",
        )
        self.strict = strict
        self._processor = None
        self._model = None
        self._model_unavailable = False
        self._fallback_dummy = False
        self._load_error_type: str | None = None
        self._expected_sampling_rate: int | None = None
        self._max_duration_sec = float(os.environ.get("KUCHIKAE_AUDIO_EMOTION_MAX_SECONDS", "15"))

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification  # type: ignore

            self._processor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self._model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            self._expected_sampling_rate = int(
                getattr(self._processor, "sampling_rate", None)
                or getattr(getattr(self._model, "config", None), "sampling_rate", None)
                or 16000
            )
        except Exception as e:
            self._model_unavailable = True
            self._load_error_type = type(e).__name__
            if self.strict:
                raise RuntimeError(
                    f"Audio emotion model {self.model_id!r} is unavailable."
                ) from e
            logger.warning("audio emotion model unavailable, using dummy behavior: %s", e)
            self._processor = None
            self._model = None

    def _resample_audio(self, samples, input_sr: int, target_sr: int):
        if input_sr == target_sr:
            return samples
        import numpy as np

        if samples.size == 0:
            return samples
        duration = samples.shape[0] / float(input_sr)
        target_len = max(1, int(round(duration * target_sr)))
        old_x = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
        new_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
        return np.interp(new_x, old_x, samples).astype("float32")

    def _prepare_audio(self, audio_path: str):
        import numpy as np
        import soundfile as sf

        samples, sr = sf.read(audio_path, dtype="float32")
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        expected_sr = self._expected_sampling_rate or sr or 16000
        samples = self._resample_audio(samples, sr, expected_sr)
        max_samples = int(expected_sr * self._max_duration_sec)
        if samples.shape[0] > max_samples:
            samples = samples[:max_samples]
        return np.asarray(samples, dtype="float32"), expected_sr

    def _map_label(self, label: str) -> tuple[str, str, float, float]:
        label = label.lower()
        if any(k in label for k in ("happy", "joy", "excited")):
            return "bright", "high", 0.8, 0.7
        if any(k in label for k in ("anger", "angry", "ang")):
            return "serious", "high", 0.85, -0.7
        if "sad" in label:
            return "calm", "low", 0.2, -0.6
        if any(k in label for k in ("neutral", "calm")):
            return "neutral", "medium", 0.4, 0.0
        return "neutral", "medium", 0.5, 0.0

    def detect(self, audio_path: str) -> AudioEmotion:
        self._load()
        if self._processor is None or self._model is None:
            self._fallback_dummy = True
            return AudioEmotion(source="fallback_dummy")
        try:
            import torch

            samples, sr = self._prepare_audio(audio_path)
            inputs = self._processor(samples, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = logits.softmax(dim=-1)[0]
            idx = int(probs.argmax().item())
            label = str(getattr(self._model.config, "id2label", {}).get(idx, str(idx))).lower()
            confidence = float(probs[idx].item())
            mood, energy, arousal, valence = self._map_label(label)
            return AudioEmotion(
                mood=mood,
                energy=energy,
                arousal=arousal,
                valence=valence,
                confidence=confidence,
                source=f"transformers:{self.model_id}",
            )
        except Exception as e:
            self._fallback_dummy = True
            self._load_error_type = type(e).__name__
            if self.strict:
                raise RuntimeError("Audio emotion inference failed.") from e
            logger.warning("audio emotion inference failed, using dummy behavior: %s", e)
            return AudioEmotion(source="fallback_dummy")

    @property
    def model_unavailable(self) -> bool:
        return self._model_unavailable

    @property
    def fallback_dummy(self) -> bool:
        return self._fallback_dummy

    @property
    def load_error_type(self) -> str | None:
        return self._load_error_type

    @property
    def expected_sampling_rate(self) -> int | None:
        return self._expected_sampling_rate
