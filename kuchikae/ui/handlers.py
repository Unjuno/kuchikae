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

    if isinstance(audio_input, tuple):
        sr, data = audio_input
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, data, sr)
            logger.info("[normalize] tuple -> %s", tmp.name)
            return tmp.name

    if isinstance(audio_input, dict):
        orig_name = audio_input.get("orig_name")
        if orig_name and os.path.isfile(orig_name):
            logger.info("[normalize] dict.orig_name -> %s", orig_name)
            return str(orig_name)
        path = audio_input.get("path")
        if path and os.path.isfile(path):
            logger.info("[normalize] dict.path -> %s", path)
            return str(path)
        name = audio_input.get("name")
        if name and os.path.isfile(name):
            logger.info("[normalize] dict.name -> %s", name)
            return str(name)
        logger.warning("[normalize] dict but no valid file. keys=%s", list(audio_input.keys()))
        return None

    path = getattr(audio_input, "path", None)
    if path and os.path.isfile(path):
        logger.info("[normalize] obj.path -> %s", path)
        return str(path)
    orig_name = getattr(audio_input, "orig_name", None)
    if orig_name and os.path.isfile(orig_name):
        logger.info("[normalize] obj.orig_name -> %s", orig_name)
        return str(orig_name)

    result = str(audio_input)
    if os.path.isfile(result):
        logger.info("[normalize] str -> %s", result)
        return result

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
    if path is None:
        logger.warning("[run_simple] no path, aborting")
        yield gr.update(), "", "", ""
        return

    prompt = TextTransformPrompt(instruction=TEMPLATES["自然に"])
    voice_prompt = normalize_voice_output_prompt(voice_output_prompt)
    stream_fn = pipeline.process_stream_live if live_streaming else pipeline.process_stream
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
                yield aud, src, txt, "言い直しました"
            elif status == "VOX":
                yield gr.update(value=None), src, txt, "変換中..."
            elif status == "TXT":
                yield gr.update(value=None), src, txt, "変換中..."
            elif status == "STT_PARTIAL":
                yield gr.update(value=None), src, "", "文字起こし中..."
            else:
                yield gr.update(value=None), "", "", "音声認識中..."
    except Exception:
        logger.exception("[run_simple] inference failed")
        raise


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
    if path is None:
        logger.error("[run] no path! audio_input type=%s", type(audio_input).__name__)
        raise gr.Error("音声を録音またはアップロードしてください")

    if template_name == "カスタム" and custom_prompt.strip():
        prompt_text = custom_prompt
    elif template_name in TEMPLATES:
        prompt_text = TEMPLATES[template_name]
    else:
        prompt_text = TEMPLATES["自然に"]

    prompt = TextTransformPrompt(instruction=prompt_text)
    voice_prompt = normalize_voice_output_prompt(voice_output_prompt)
    stream_fn = pipeline.process_stream_live if live_streaming else pipeline.process_stream
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
                yield aud, src, txt, "言い直しました"
            elif status == "VOX":
                yield gr.update(value=None), src, txt, "変換中..."
            elif status == "TXT":
                yield gr.update(value=None), src, txt, "変換中..."
            elif status == "STT_PARTIAL":
                yield gr.update(value=None), src, "", "文字起こし中..."
            else:
                yield gr.update(value=None), "", "", "音声認識中..."
    except Exception:
        logger.exception("[run] inference failed")
        raise


def on_template_change(template_name: str):
    if template_name == "カスタム":
        return gr.update()
    text = TEMPLATES.get(template_name, "")
    return gr.update(value=text)
