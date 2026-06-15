"""Real voice output backends (Irodori TTS, OpenVoice)."""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from kuchikae.domain.voice_output import VoiceOutputBackend
from kuchikae.domain.types import VoiceContext, VoiceOutputPrompt

logger = logging.getLogger(__name__)
OUTPUT_DIR = "outputs"


class OpenVoiceOutputBackend(VoiceOutputBackend):

    def __init__(self, openvoice_path: str | None = None) -> None:
        self._openvoice_path = openvoice_path or os.environ.get(
            "KUCHIKAE_OPENVOICE_PATH", ""
        )
        self._base_tts: Any = None
        self._converter: Any = None

    def _log(self, msg: str) -> None:
        logger.info("[OpenVoice] %s", msg)

    def _ensure_models_loaded(self) -> tuple[Any, Any]:
        if self._base_tts is not None and self._converter is not None:
            return self._base_tts, self._converter

        if not self._openvoice_path:
            raise RuntimeError(
                "OpenVoice path is not configured. "
                "Set KUCHIKAE_OPENVOICE_PATH environment variable to the path of your OpenVoice checkout."
            )

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
        import openvoice.se_extractor as se_extractor_module

        # The previous implementation re-used the same path repeatedly and did
        # not actually diversify chunks. Extract once until a real chunked
        # prosody path is implemented.
        return se_extractor_module.extract_se(
            audio_path,
            device="cpu",
            fsave_se=False,
            save_path=OUTPUT_DIR,
        )

    def synthesize(
        self,
        text: str,
        voice_context: VoiceContext,
        prompt: VoiceOutputPrompt | None = None,
    ) -> str:
        if not voice_context.ready or not voice_context.reference_audio_path:
            raise RuntimeError("OpenVoice requires a ready reference audio path.")

        if prompt is not None:
            logger.warning("[OpenVoice] prompt.instruction is ignored (OpenVoice API has no caption/emotion control)")

        t0 = time.time()
        base_tts, converter = self._ensure_models_loaded()
        self._log(f"model load: {time.time()-t0:.2f}s")

        target_se = self._extract_multi_frame_se(voice_context.reference_audio_path)
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
        num_steps: int = 6,
        cfg_scale_text: float = 2.0,
        cfg_scale_speaker: float = 3.0,
        cfg_guidance_mode: str = "independent",
        cfg_scale: float | None = None,
        cfg_min_t: float = 0.5,
        cfg_max_t: float = 1.0,
        speaker_kv_scale: float | None = None,
        speaker_kv_min_t: float | None = None,
        speaker_kv_max_layers: int | None = None,
        speaker_uncond_mode: str = "mask",
        t_schedule_mode: str = "linear",
        sway_coeff: float = -1.0,
        ref_normalize_db: float = -16.0,
        ref_ensure_max: bool = True,
        decode_mode: str = "sequential",
        duration_scale: float = 1.0,
        min_seconds: float = 0.5,
        max_seconds: float = 30.0,
        max_ref_seconds: float | None = 30.0,
        max_text_len: int | None = None,
        max_caption_len: int | None = None,
        truncation_factor: float | None = None,
        rescale_k: float | None = None,
        rescale_sigma: float | None = None,
        tail_window_size: int = 20,
        tail_std_threshold: float = 0.05,
        tail_mean_threshold: float = 0.1,
        lora_adapter: str | None = None,
    ) -> None:
        self._hf_checkpoint = hf_checkpoint
        self._codec_repo = codec_repo
        self._num_steps = num_steps
        self._cfg_scale_text = cfg_scale_text
        self._cfg_scale_speaker = cfg_scale_speaker
        self._cfg_guidance_mode = cfg_guidance_mode
        self._cfg_scale = cfg_scale
        self._cfg_min_t = cfg_min_t
        self._cfg_max_t = cfg_max_t
        self._speaker_kv_scale = speaker_kv_scale
        self._speaker_kv_min_t = speaker_kv_min_t
        self._speaker_kv_max_layers = speaker_kv_max_layers
        self._speaker_uncond_mode = speaker_uncond_mode
        self._t_schedule_mode = t_schedule_mode
        self._sway_coeff = sway_coeff
        self._ref_normalize_db = ref_normalize_db
        self._ref_ensure_max = ref_ensure_max
        self._decode_mode = decode_mode
        self._duration_scale = duration_scale
        self._min_seconds = min_seconds
        self._max_seconds = max_seconds
        self._max_ref_seconds = max_ref_seconds
        self._max_text_len = max_text_len
        self._max_caption_len = max_caption_len
        self._truncation_factor = truncation_factor
        self._rescale_k = rescale_k
        self._rescale_sigma = rescale_sigma
        self._tail_window_size = tail_window_size
        self._tail_std_threshold = tail_std_threshold
        self._tail_mean_threshold = tail_mean_threshold
        self._lora_adapter = lora_adapter
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
        self._hf_checkpoint = os.environ.get("IRODORI_MODEL_ID", self._hf_checkpoint)
        self._codec_repo = os.environ.get("IRODORI_CODEC_REPO", self._codec_repo)
        self._num_steps = int(os.environ.get("IRODORI_NUM_STEPS", str(self._num_steps)))
        self._cfg_scale_text = float(os.environ.get("IRODORI_CFG_SCALE_TEXT", str(self._cfg_scale_text)))
        self._cfg_scale_speaker = float(os.environ.get("IRODORI_CFG_SCALE_SPEAKER", str(self._cfg_scale_speaker)))
        self._cfg_scale_caption = (
            float(os.environ["IRODORI_CFG_SCALE_CAPTION"]) if os.environ.get("IRODORI_CFG_SCALE_CAPTION") else self._cfg_scale_text
        )
        self._cfg_guidance_mode = os.environ.get("IRODORI_CFG_GUIDANCE_MODE", self._cfg_guidance_mode)
        self._cfg_scale = (
            float(os.environ["IRODORI_CFG_SCALE"]) if os.environ.get("IRODORI_CFG_SCALE") else self._cfg_scale
        )
        self._cfg_min_t = float(os.environ.get("IRODORI_CFG_MIN_T", str(self._cfg_min_t)))
        self._cfg_max_t = float(os.environ.get("IRODORI_CFG_MAX_T", str(self._cfg_max_t)))
        self._speaker_kv_scale = (
            float(os.environ["IRODORI_SPEAKER_KV_SCALE"]) if os.environ.get("IRODORI_SPEAKER_KV_SCALE") else self._speaker_kv_scale
        )
        self._speaker_kv_min_t = (
            float(os.environ["IRODORI_SPEAKER_KV_MIN_T"]) if os.environ.get("IRODORI_SPEAKER_KV_MIN_T") else self._speaker_kv_min_t
        )
        self._speaker_kv_max_layers = (
            int(os.environ["IRODORI_SPEAKER_KV_MAX_LAYERS"]) if os.environ.get("IRODORI_SPEAKER_KV_MAX_LAYERS") else self._speaker_kv_max_layers
        )
        self._speaker_uncond_mode = os.environ.get("IRODORI_SPEAKER_UNCOND_MODE", self._speaker_uncond_mode)
        self._t_schedule_mode = os.environ.get("IRODORI_T_SCHEDULE_MODE", self._t_schedule_mode)
        self._sway_coeff = float(os.environ.get("IRODORI_SWAY_COEFF", str(self._sway_coeff)))
        self._ref_normalize_db = float(os.environ.get("IRODORI_REF_NORMALIZE_DB", str(self._ref_normalize_db)))
        self._ref_ensure_max = os.environ.get("IRODORI_REF_ENSURE_MAX", str(int(self._ref_ensure_max))).lower() in ("1", "true", "yes")
        self._decode_mode = os.environ.get("IRODORI_DECODE_MODE", self._decode_mode)
        self._duration_scale = float(os.environ.get("IRODORI_DURATION_SCALE", str(self._duration_scale)))
        self._min_seconds = float(os.environ.get("IRODORI_MIN_SECONDS", str(self._min_seconds)))
        self._max_seconds = float(os.environ.get("IRODORI_MAX_SECONDS", str(self._max_seconds)))
        self._max_ref_seconds = (
            float(os.environ["IRODORI_MAX_REF_SECONDS"]) if os.environ.get("IRODORI_MAX_REF_SECONDS") else self._max_ref_seconds
        )
        self._max_text_len = int(os.environ["IRODORI_MAX_TEXT_LEN"]) if os.environ.get("IRODORI_MAX_TEXT_LEN") else self._max_text_len
        self._max_caption_len = int(os.environ["IRODORI_MAX_CAPTION_LEN"]) if os.environ.get("IRODORI_MAX_CAPTION_LEN") else self._max_caption_len
        self._truncation_factor = float(os.environ["IRODORI_TRUNCATION_FACTOR"]) if os.environ.get("IRODORI_TRUNCATION_FACTOR") else self._truncation_factor
        self._rescale_k = float(os.environ["IRODORI_RESCALE_K"]) if os.environ.get("IRODORI_RESCALE_K") else self._rescale_k
        self._rescale_sigma = float(os.environ["IRODORI_RESCALE_SIGMA"]) if os.environ.get("IRODORI_RESCALE_SIGMA") else self._rescale_sigma
        self._tail_window_size = int(os.environ.get("IRODORI_TAIL_WINDOW_SIZE", str(self._tail_window_size)))
        self._tail_std_threshold = float(os.environ.get("IRODORI_TAIL_STD_THRESHOLD", str(self._tail_std_threshold)))
        self._tail_mean_threshold = float(os.environ.get("IRODORI_TAIL_MEAN_THRESHOLD", str(self._tail_mean_threshold)))
        self._lora_adapter = os.environ.get("IRODORI_LORA_ADAPTER", self._lora_adapter)
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

    def synthesize(
        self,
        text: str,
        voice_context: VoiceContext,
        prompt: VoiceOutputPrompt | None = None,
    ) -> str:
        if not voice_context.ready or not voice_context.reference_audio_path:
            raise RuntimeError("Irodori-TTS requires a ready reference audio path.")

        stripped = text.strip()
        if not stripped:
            raise RuntimeError("Irodori-TTS requires non-empty text.")

        t0 = time.time()
        runtime = self._ensure_runtime()
        self._log(f"runtime ready: {time.time()-t0:.2f}s")

        from irodori_tts.inference_runtime import SamplingRequest, save_wav

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(
            OUTPUT_DIR,
            f"irodori_output_{int(time.time() * 1000)}_{os.getpid()}.wav",
        )

        caption = prompt.instruction if prompt else None
        self._log(f"synthesizing ({self._num_steps} steps, linear)... caption={repr(caption[:80]) if caption else None}")
        t1 = time.time()
        result = runtime.synthesize(
            SamplingRequest(
                text=stripped,
                caption=caption,
                ref_wav=voice_context.reference_audio_path,
                decode_mode=self._decode_mode,
                duration_scale=self._duration_scale,
                min_seconds=self._min_seconds,
                max_seconds=self._max_seconds,
                max_ref_seconds=self._max_ref_seconds,
                max_text_len=self._max_text_len,
                max_caption_len=self._max_caption_len,
                num_steps=self._num_steps,
                cfg_guidance_mode=self._cfg_guidance_mode,
                cfg_scale=self._cfg_scale,
                t_schedule_mode=self._t_schedule_mode,
                sway_coeff=self._sway_coeff,
                ref_normalize_db=self._ref_normalize_db,
                ref_ensure_max=self._ref_ensure_max,
                cfg_scale_text=self._cfg_scale_text,
                cfg_scale_caption=self._cfg_scale_caption,
                cfg_scale_speaker=self._cfg_scale_speaker,
                cfg_min_t=self._cfg_min_t,
                cfg_max_t=self._cfg_max_t,
                truncation_factor=self._truncation_factor,
                rescale_k=self._rescale_k,
                rescale_sigma=self._rescale_sigma,
                trim_tail=True,
                tail_window_size=self._tail_window_size,
                tail_std_threshold=self._tail_std_threshold,
                tail_mean_threshold=self._tail_mean_threshold,
                context_kv_cache=True,
                speaker_kv_scale=self._speaker_kv_scale,
                speaker_kv_min_t=self._speaker_kv_min_t,
                speaker_kv_max_layers=self._speaker_kv_max_layers,
                speaker_uncond_mode=self._speaker_uncond_mode,
                lora_adapter=self._lora_adapter,
            )
        )
        self._log(f"inference: {time.time()-t1:.2f}s")

        saved = save_wav(output_path, result.audio, result.sample_rate)
        self._log(f"saved: {os.path.getsize(output_path)/1e6:.1f}MB")
        return str(saved)
