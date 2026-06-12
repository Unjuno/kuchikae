from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Protocol

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

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        try:
            from transformers import AutoProcessor, AutoModelForAudioClassification  # type: ignore

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForAudioClassification.from_pretrained(self.model_id)
        except Exception as e:
            if self.strict:
                raise RuntimeError(
                    f"Audio emotion model {self.model_id!r} is unavailable."
                ) from e
            logger.warning("audio emotion model unavailable, using dummy behavior: %s", e)
            self._processor = None
            self._model = None

    def detect(self, audio_path: str) -> AudioEmotion:
        self._load()
        if self._processor is None or self._model is None:
            return AudioEmotion()
        try:
            import numpy as np
            import soundfile as sf
            import torch

            samples, sr = sf.read(audio_path, dtype="float32")
            if samples.ndim > 1:
                samples = np.mean(samples, axis=1)
            inputs = self._processor(samples, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = logits.softmax(dim=-1)[0]
            idx = int(probs.argmax().item())
            label = str(getattr(self._model.config, "id2label", {}).get(idx, str(idx))).lower()
            confidence = float(probs[idx].item())
            mood = "neutral"
            energy = "medium"
            arousal = 0.5
            valence = 0.0
            if any(k in label for k in ("ang", "anger", "excited", "happy", "joy")):
                energy = "high"
                arousal = 0.8
            elif any(k in label for k in ("sad", "calm", "neutral", "bored")):
                energy = "low" if "sad" in label or "bored" in label else "medium"
                arousal = 0.2 if energy == "low" else 0.4
            if "happy" in label or "joy" in label:
                valence = 0.7
            elif "sad" in label or "ang" in label:
                valence = -0.6
            return AudioEmotion(
                mood=mood,
                energy=energy,
                arousal=arousal,
                valence=valence,
                confidence=confidence,
                source=f"transformers:{self.model_id}",
            )
        except Exception as e:
            if self.strict:
                raise RuntimeError("Audio emotion inference failed.") from e
            logger.warning("audio emotion inference failed, using dummy behavior: %s", e)
            return AudioEmotion()
