"""Kuchikae UI — event handlers and helpers.

All module-level functions for testability (View Isolation).
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Generator

import gradio as gr
import soundfile as sf

from kuchikae.domain.types import TextTransformPrompt, VoiceOutputPrompt
from kuchikae.pipeline import KuchikaePipeline

logger = logging.getLogger("kuchikae.ui.handlers")

TEMPLATES = {
    "自然に": "内容、数字、日時、固有名詞、否定条件は保ちつつ、言い回しを自然な日本語に変換してください。",
    "丁寧に": "次のテキストを「です・ます」調の丁寧な言葉遣いに変換してください。",
    "柔らかく": "次のテキストを柔らかく丁寧な表現に変換してください。",
    "短く": "次のテキストを簡潔に短く要約・変換してください。内容や固有名詞は変えないでください。",
    "カスタム": "",
}


def normalize_audio_path(audio_input) -> str | None:
    if audio_input is None:
        logger.warning("[normalize] input is None")
        return None

    def _valid_local_file(candidate) -> str | None:
        if not candidate:
            return None
        path = str(candidate).strip()
        if path and os.path.isfile(path):
            return path
        return None

    if isinstance(audio_input, tuple):
        if len(audio_input) == 2 and isinstance(audio_input[0], (int, float)) and hasattr(audio_input[1], "__array__"):
            sr, data = audio_input
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, data, sr)
                logger.info("[normalize] tuple(sr,data) -> %s", tmp.name)
                return tmp.name
        if len(audio_input) >= 1 and isinstance(audio_input[0], str) and audio_input[0]:
            path = _valid_local_file(audio_input[0])
            if path:
                logger.info("[normalize] tuple[0] -> %s", path)
                return path
        if len(audio_input) == 2:
            sr, data = audio_input
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, data, sr)
                logger.info("[normalize] tuple -> %s", tmp.name)
                return tmp.name
        logger.warning("[normalize] tuple but unsupported shape=%s", len(audio_input))
        return None

    if isinstance(audio_input, dict):
        for key in ("path", "name", "orig_name", "filepath"):
            candidate = audio_input.get(key)
            path = _valid_local_file(candidate)
            if path:
                logger.info("[normalize] dict.%s -> %s", key, path)
                return path
        logger.warning(
            "[normalize] dict but no valid local file. keys=%s repr=%r",
            list(audio_input.keys()),
            audio_input,
        )
        return None

    for attr in ("path", "name", "orig_name", "filepath"):
        candidate = getattr(audio_input, attr, None)
        path = _valid_local_file(candidate)
        if path:
            logger.info("[normalize] obj.%s -> %s", attr, path)
            return path

    if isinstance(audio_input, str):
        path = _valid_local_file(audio_input)
        if path:
            logger.info("[normalize] str -> %s", path)
            return path
        logger.warning("[normalize] str but not a local file: %r", audio_input)
        return None

    logger.warning("[normalize] unresolvable: type=%s repr=%s", type(audio_input).__name__, repr(audio_input)[:200])
    return None


def normalize_voice_output_prompt(value) -> VoiceOutputPrompt | None:
    if value is None:
        return None
    if isinstance(value, VoiceOutputPrompt):
        return value
    text = str(value).strip()
    if not text:
        return None
    return VoiceOutputPrompt(instruction=text)


def _backend_status(pipeline: KuchikaePipeline) -> str:
    cache_state = "disabled" if getattr(pipeline, "disable_processing_cache", False) else "enabled"
    summary = ""
    if hasattr(pipeline, "diagnostics") and pipeline.diagnostics is not None:
        summary = pipeline.diagnostics.user_summary().replace("\n", " | ")
    stt_backend = type(getattr(pipeline, "stt_backend", object())).__name__
    text_backend = type(getattr(pipeline, "text_transform_backend", object())).__name__
    voice_backend = type(getattr(pipeline, "voice_output_backend", object())).__name__
    base = (
        f"STT: {stt_backend} | "
        f"TEXT: {text_backend} | "
        f"VOICE: {voice_backend} | "
        f"CACHE: {cache_state}"
    )
    return f"{base} | {summary}" if summary else base


def run_simple(
    pipeline: KuchikaePipeline,
    audio_input,
    live_streaming: bool = False,
    voice_output_prompt=None,
) -> Generator:
    logger.info(
        "[run_simple] called live_streaming=%s audio_type=%s audio_repr=%r voice_prompt=%s",
        live_streaming,
        type(audio_input).__name__,
        repr(audio_input)[:400],
        "set" if voice_output_prompt is not None else "none",
    )
    path = normalize_audio_path(audio_input)
    last_status = "音声認識中"
    last_source = ""
    last_text = ""
    if path is None:
        logger.warning("[run_simple] no path, aborting")
        yield (
            gr.update(value=None),
            "",
            "",
            "録音ファイルを取得できませんでした。マイク権限を許可し、もう一度押して話してください。",
        )
        return

    prompt = TextTransformPrompt(instruction=TEMPLATES["自然に"])
    voice_prompt = normalize_voice_output_prompt(voice_output_prompt)
    stream_fn = pipeline.process_stream_live if live_streaming else pipeline.process_stream
    backend_status = _backend_status(pipeline)
    logger.info(
        "[run_simple] stream_fn=%s path=%s text_prompt_preview=%r voice_prompt=%s",
        getattr(stream_fn, "__name__", repr(stream_fn)),
        path,
        prompt.instruction[:160],
        "set" if voice_prompt is not None else "none",
    )

    try:
        for idx, (status, src, txt, aud) in enumerate(
            stream_fn(path, prompt, voice_prompt),
            start=1,
        ):
            last_status = status
            last_source = src or ""
            last_text = txt or ""
            logger.info(
                "[run_simple] yield idx=%d status=%s src_len=%s txt_len=%s aud=%s src_preview=%r txt_preview=%r",
                idx,
                status,
                len(src) if isinstance(src, str) else None,
                len(txt) if isinstance(txt, str) else None,
                aud,
                src[:120] if isinstance(src, str) else src,
                txt[:120] if isinstance(txt, str) else txt,
            )
            if status == "DONE":
                yield aud, src, txt, f"言い直しました | {backend_status}"
            elif status == "VOX":
                yield gr.update(value=None), src, txt, f"変換中... | {backend_status}"
            elif status == "TXT":
                yield gr.update(value=None), src, txt, f"変換中... | {backend_status}"
            elif status == "STT_PARTIAL":
                yield gr.update(value=None), src, "", f"文字起こし中... | {backend_status}"
            else:
                yield gr.update(value=None), "", "", f"音声認識中... | {backend_status}"
    except Exception as e:
        logger.exception("[run_simple] inference failed")
        yield (
            gr.update(value=None),
            last_source,
            last_text,
            f"{last_status} 段階で失敗しました: {type(e).__name__}: {e}",
        )
        return


def run(
    audio_input,
    template_name: str,
    custom_prompt: str,
    pipeline: KuchikaePipeline,
    live_streaming: bool = False,
    voice_output_prompt=None,
) -> Generator:
    logger.info(
        "[run] called template=%s live_streaming=%s audio_type=%s audio_repr=%r voice_prompt=%s",
        template_name,
        live_streaming,
        type(audio_input).__name__,
        repr(audio_input)[:400],
        "set" if voice_output_prompt is not None else "none",
    )
    path = normalize_audio_path(audio_input)
    last_status = "音声認識中"
    last_source = ""
    last_text = ""
    if path is None:
        logger.error("[run] no path! audio_input type=%s", type(audio_input).__name__)
        yield (
            gr.update(value=None),
            "",
            "",
            f"AudioInput 段階で失敗しました: ValueError: 音声を録音してください。必要なら通常モードで録音し直してください。 (audio_input_type={type(audio_input).__name__})",
        )
        return

    if template_name == "カスタム" and custom_prompt.strip():
        prompt_text = custom_prompt
    elif template_name in TEMPLATES:
        prompt_text = TEMPLATES[template_name]
    else:
        prompt_text = TEMPLATES["自然に"]

    prompt = TextTransformPrompt(instruction=prompt_text)
    voice_prompt = normalize_voice_output_prompt(voice_output_prompt)
    stream_fn = pipeline.process_stream_live if live_streaming else pipeline.process_stream
    backend_status = _backend_status(pipeline)
    logger.info(
        "[run] stream_fn=%s path=%s text_prompt_preview=%r voice_prompt=%s",
        getattr(stream_fn, "__name__", repr(stream_fn)),
        path,
        prompt.instruction[:160],
        "set" if voice_prompt is not None else "none",
    )

    try:
        for idx, (status, src, txt, aud) in enumerate(
            stream_fn(path, prompt, voice_prompt),
            start=1,
        ):
            last_status = status
            last_source = src or ""
            last_text = txt or ""
            logger.info(
                "[run] yield idx=%d status=%s src_len=%s txt_len=%s aud=%s src_preview=%r txt_preview=%r",
                idx,
                status,
                len(src) if isinstance(src, str) else None,
                len(txt) if isinstance(txt, str) else None,
                aud,
                src[:120] if isinstance(src, str) else src,
                txt[:120] if isinstance(txt, str) else txt,
            )
            if status == "DONE":
                yield aud, src, txt, f"言い直しました | {backend_status}"
            elif status == "VOX":
                yield gr.update(value=None), src, txt, f"変換中... | {backend_status}"
            elif status == "TXT":
                yield gr.update(value=None), src, txt, f"変換中... | {backend_status}"
            elif status == "STT_PARTIAL":
                yield gr.update(value=None), src, "", f"文字起こし中... | {backend_status}"
            else:
                yield gr.update(value=None), "", "", f"音声認識中... | {backend_status}"
    except Exception as e:
        logger.exception("[run] inference failed")
        yield (
            gr.update(value=None),
            last_source,
            last_text,
            f"{last_status} 段階で失敗しました: {type(e).__name__}: {e}",
        )
        return


def on_template_change(template_name: str):
    if template_name == "カスタム":
        return gr.update()
    text = TEMPLATES.get(template_name, "")
    return gr.update(value=text)
