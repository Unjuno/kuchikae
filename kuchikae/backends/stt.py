"""Real STT backends (FasterWhisper)."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Generator

import soundfile as sf

from kuchikae.domain.audio_stream import AudioChunk
from kuchikae.domain.stt import FasterWhisperConfig, STTBackend, StreamingSTTBackend
from kuchikae.domain.types import STTPartial

logger = logging.getLogger(__name__)


class FasterWhisperSTTBackend(STTBackend):

    def __init__(
        self,
        config: FasterWhisperConfig | None = None,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
        beam_size: int | None = None,
        vad_filter: bool | None = None,
        temperature: float | None = None,
        condition_on_previous_text: bool | None = None,
    ) -> None:
        resolved = config or FasterWhisperConfig()
        self._config = FasterWhisperConfig(
            model_size=model_size or resolved.model_size,
            device=device or resolved.device,
            compute_type=compute_type or resolved.compute_type,
            language=language or resolved.language,
            beam_size=beam_size if beam_size is not None else resolved.beam_size,
            vad_filter=vad_filter if vad_filter is not None else resolved.vad_filter,
            temperature=temperature if temperature is not None else resolved.temperature,
            condition_on_previous_text=(
                condition_on_previous_text
                if condition_on_previous_text is not None
                else resolved.condition_on_previous_text
            ),
        )
        self._model = None
        self._resolved_model_size: str | None = None
        self._resolved_device: str | None = None
        self._resolved_compute_type: str | None = None
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "FasterWhisperSTTBackend requires the ``faster-whisper`` package. "
                "Install with ``uv pip install faster-whisper``."
            )

    @property
    def config(self) -> FasterWhisperConfig:
        return self._config

    @property
    def model_size(self) -> str:
        return self._resolved_model_size or self._config.model_size

    @property
    def device(self) -> str:
        return self._resolved_device or self._config.device

    @property
    def compute_type(self) -> str:
        return self._resolved_compute_type or self._config.compute_type

    def _load_model(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        model_size = os.environ.get("WHISPER_MODEL_SIZE", self._config.model_size)
        device = os.environ.get("WHISPER_DEVICE", self._config.device)
        compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", self._config.compute_type)
        logger.info("loading whisper model '%s' (device=%s compute_type=%s)...", model_size, device, compute_type)
        t0 = time.time()
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._resolved_model_size = model_size
        self._resolved_device = device
        self._resolved_compute_type = compute_type
        logger.info("whisper model loaded in %.2fs", time.time() - t0)
        return self._model

    def transcribe(self, audio_path: str) -> str:
        model = self._load_model()

        t1 = time.time()
        segments, info = model.transcribe(
            audio_path,
            language=self._config.language,
            beam_size=int(os.environ.get("WHISPER_BEAM_SIZE", str(self._config.beam_size))),
            vad_filter=os.environ.get("WHISPER_VAD_FILTER", str(int(self._config.vad_filter))).lower() in ("1", "true", "yes"),
            temperature=float(os.environ.get("WHISPER_TEMPERATURE", str(self._config.temperature))),
            condition_on_previous_text=os.environ.get(
                "WHISPER_CONDITION_ON_PREVIOUS_TEXT", str(int(self._config.condition_on_previous_text))
            ).lower() in ("1", "true", "yes"),
        )
        logger.info("whisper transcribe: %.2fs", time.time() - t1)

        result = " ".join(seg.text for seg in segments)
        logger.info("whisper result: %s", result[:80])
        return result

    def transcribe_stream(self, audio_path: str) -> Generator[str, None, None]:
        model = self._load_model()

        t1 = time.time()
        segments, info = model.transcribe(
            audio_path,
            language=self._config.language,
            beam_size=int(os.environ.get("WHISPER_BEAM_SIZE", str(self._config.beam_size))),
            vad_filter=os.environ.get("WHISPER_VAD_FILTER", str(int(self._config.vad_filter))).lower() in ("1", "true", "yes"),
            temperature=float(os.environ.get("WHISPER_TEMPERATURE", str(self._config.temperature))),
            condition_on_previous_text=os.environ.get(
                "WHISPER_CONDITION_ON_PREVIOUS_TEXT", str(int(self._config.condition_on_previous_text))
            ).lower() in ("1", "true", "yes"),
        )
        logger.info("whisper transcribe: %.2fs", time.time() - t1)

        accumulated = []
        for seg in segments:
            accumulated.append(seg.text)
            yield " ".join(accumulated)


class StreamingFasterWhisperSTTBackend(FasterWhisperSTTBackend):
    """Streaming STT using fixed-window chunking for push-to-talk.
    
    Processes audio in small chunks and yields partial transcripts.
    Suitable for showing live transcription during/after recording.
    """

    def __init__(self, model_size: str = "tiny", chunk_sec: float = 5.0, overlap_sec: float = 1.0) -> None:
        super().__init__(model_size=model_size)
        self._chunk_sec = chunk_sec
        self._overlap_sec = overlap_sec

    def transcribe_stream(self, audio_path: str) -> Generator[str, None, None]:
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        chunk_samples = int(self._chunk_sec * sr)
        overlap_samples = int(self._overlap_sec * sr)
        stride = chunk_samples - overlap_samples
        total = len(data)
        
        model = self._load_model()
        accumulated: list[str] = []
        
        start = 0
        while start < total:
            end = min(start + chunk_samples, total)
            chunk = data[start:end]
            
            tmp_name = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_name = tmp.name
                    sf.write(tmp.name, chunk, sr)
                segments, _ = model.transcribe(
                    tmp_name,
                    language=self._config.language,
                    beam_size=int(os.environ.get("WHISPER_BEAM_SIZE", str(self._config.beam_size))),
                    vad_filter=os.environ.get("WHISPER_VAD_FILTER", str(int(self._config.vad_filter))).lower() in ("1", "true", "yes"),
                    temperature=float(os.environ.get("WHISPER_TEMPERATURE", str(self._config.temperature))),
                    condition_on_previous_text=os.environ.get(
                        "WHISPER_CONDITION_ON_PREVIOUS_TEXT", str(int(self._config.condition_on_previous_text))
                    ).lower() in ("1", "true", "yes"),
                )
            finally:
                if tmp_name and os.path.exists(tmp_name):
                    os.unlink(tmp_name)

            chunk_text = " ".join(seg.text for seg in segments).strip()
            if chunk_text:
                if accumulated and chunk_text.startswith(accumulated[-1]):
                    accumulated[-1] = chunk_text
                elif accumulated and accumulated[-1] and accumulated[-1] in chunk_text:
                    accumulated[-1] = chunk_text
                else:
                    accumulated.append(chunk_text)
                yield " ".join(accumulated)
            
            if end >= total:
                break
            start += stride


class ChunkedStreamingSTTBackend(StreamingSTTBackend):
    """Real streaming STT that uses AudioChunker + FasterWhisperSTTBackend.

    Accumulates chunks and transcribes on each push.
    Uses simple stable-prefix heuristic: the text that hasn't changed
    across the last N iterations is considered stable.
    """

    def __init__(
        self,
        inner: FasterWhisperSTTBackend | None = None,
        stable_window: int = 2,
    ) -> None:
        self._inner = inner or FasterWhisperSTTBackend(model_size="tiny")
        self._stable_window = stable_window
        self._chunk_texts: list[str] = []

    def push_audio(self, chunk: AudioChunk) -> STTPartial:
        import tempfile

        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
                sf.write(tmp.name, chunk.samples, chunk.sample_rate)
            text = self._inner.transcribe(tmp.name)
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)

        self._chunk_texts.append(text)
        full_text = " ".join(self._chunk_texts)

        stable_prefix = ""
        last_n = self._chunk_texts[-self._stable_window:]
        if len(last_n) >= 2:
            if len(set(last_n)) == 1:
                stable_prefix = last_n[0]
            else:
                common = self._longest_common_prefix(last_n[0], last_n[-1])
                if len(common) > 2:
                    stable_prefix = common

        unstable_suffix = full_text[len(stable_prefix):] if stable_prefix else full_text

        return STTPartial(
            session_id=chunk.session_id,
            text=full_text,
            stable_prefix=stable_prefix,
            unstable_suffix=unstable_suffix,
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
        )

    def flush(self, session_id: str) -> STTPartial | None:
        if not self._chunk_texts:
            return None
        full = " ".join(self._chunk_texts)
        return STTPartial(
            session_id=session_id,
            text=full,
            stable_prefix=full,
            unstable_suffix="",
            start_sec=0.0,
            end_sec=0.0,
        )

    @staticmethod
    def _longest_common_prefix(a: str, b: str) -> str:
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return a[:i]
