from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import sys
import types
from importlib.machinery import ModuleSpec
from kuchikae.domain.audio_emotion import AudioEmotion, DummyAudioEmotionDetector, TransformersAudioEmotionDetector


def _write_wav(path: str, sr: int = 16000, seconds: float = 1.0) -> None:
    data = np.zeros(int(sr * seconds), dtype=np.float32)
    sf.write(path, data, sr)


def test_dummy_audio_emotion_detector_returns_neutral() -> None:
    emotion = DummyAudioEmotionDetector().detect("/tmp/ignored.wav")
    assert emotion.mood == "neutral"
    assert emotion.energy == "medium"
    assert emotion.confidence == 0.0


def test_transformers_audio_emotion_resamples_and_trims(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "audio.wav"
    _write_wav(str(wav), sr=8000, seconds=20.0)

    class FakeProcessor:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []
            self.sampling_rate = 16000

        def __call__(self, samples, sampling_rate: int, return_tensors: str):
            self.calls.append((len(samples), sampling_rate))
            return {"input_values": object()}

    class FakeScalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    class FakeVector:
        def __init__(self, values: list[float]) -> None:
            self.values = values

        def softmax(self, dim: int = -1):
            total = sum(self.values)
            return [FakeVector([v / total for v in self.values])]

        def __getitem__(self, idx: int) -> FakeScalar:
            return FakeScalar(self.values[idx])

        def argmax(self):
            class _Idx:
                def __init__(self, idx: int) -> None:
                    self.idx = idx

                def item(self) -> int:
                    return self.idx

            return _Idx(max(range(len(self.values)), key=self.values.__getitem__))

    class FakeOutput:
        def __init__(self) -> None:
            self.logits = FakeVector([0.1, 0.9])

    class FakeModel:
        def __init__(self) -> None:
            self.config = type("C", (), {"id2label": {0: "neutral", 1: "happy"}})()

        def __call__(self, **kwargs):
            return FakeOutput()

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.__spec__ = ModuleSpec("torch", loader=None)
    fake_torch.float32 = "float32"
    fake_torch.no_grad = lambda: FakeNoGrad()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    processor = FakeProcessor()
    detector = TransformersAudioEmotionDetector(model_id="fake-model", strict=False)
    detector._processor = processor
    detector._model = FakeModel()
    detector._expected_sampling_rate = 16000

    emotion = detector.detect(str(wav))

    assert emotion.mood == "happy"
    assert emotion.energy == "high"
    assert emotion.valence > 0
    assert len(processor.calls) > 0
    samples_len, sr = processor.calls[-1]
    assert sr == 16000
    assert samples_len <= 16000 * 15


def test_transformers_audio_emotion_unknown_label_defaults_to_neutral(tmp_path) -> None:
    wav = tmp_path / "audio.wav"
    _write_wav(str(wav))

    class FakeProcessor:
        def __call__(self, samples, sampling_rate: int, return_tensors: str):
            return {"input_values": object()}

    class FakeVector:
        def __init__(self, values: list[float]) -> None:
            self.values = values

        def softmax(self, dim: int = -1):
            total = sum(self.values)
            return [FakeVector([v / total for v in self.values])]

        def __getitem__(self, idx: int):
            class _Scalar:
                def __init__(self, value: float) -> None:
                    self.value = value

                def item(self) -> float:
                    return self.value

            return _Scalar(self.values[idx])

        def argmax(self):
            class _Idx:
                def __init__(self, idx: int) -> None:
                    self.idx = idx

                def item(self) -> int:
                    return self.idx

            return _Idx(max(range(len(self.values)), key=self.values.__getitem__))

    class FakeOutput:
        def __init__(self) -> None:
            self.logits = FakeVector([0.9, 0.1])

    class FakeModel:
        def __init__(self) -> None:
            self.config = type("C", (), {"id2label": {0: "mystery"}})()

        def __call__(self, **kwargs):
            return FakeOutput()

    detector = TransformersAudioEmotionDetector(model_id="fake-model", strict=False)
    detector._processor = FakeProcessor()
    detector._model = FakeModel()
    detector._expected_sampling_rate = 16000

    emotion = detector.detect(str(wav))

    assert emotion.mood == "neutral"
    assert emotion.energy == "medium"


def test_transformers_audio_emotion_fallback_when_unavailable(monkeypatch, tmp_path) -> None:
    wav = tmp_path / "audio.wav"
    _write_wav(str(wav))

    detector = TransformersAudioEmotionDetector(model_id="missing-model", strict=False)

    def fake_load() -> None:
        detector._model_unavailable = True
        detector._load_error_type = "RuntimeError"
        detector._processor = None
        detector._model = None

    detector._load = fake_load  # type: ignore[method-assign]
    emotion = detector.detect(str(wav))

    assert emotion.source == "fallback_dummy"
    assert detector.model_unavailable is True
    assert detector.fallback_dummy is True
    assert detector.load_error_type in {"RuntimeError", "OSError", "ValueError", "ImportError"}


def test_transformers_audio_emotion_strict_raises(tmp_path) -> None:
    wav = tmp_path / "audio.wav"
    _write_wav(str(wav))

    detector = TransformersAudioEmotionDetector(model_id="missing-model", strict=True)

    def fake_load() -> None:
        raise RuntimeError("missing")

    detector._load = fake_load  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        detector.detect(str(wav))
