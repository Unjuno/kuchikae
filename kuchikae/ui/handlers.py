"""Kuchikae UI — event handlers and helpers.

All module-level functions for testability (View Isolation).
"""

from __future__ import annotations

import os
import tempfile
from typing import Generator

import gradio as gr
import soundfile as sf

from kuchikae.domain.types import TextTransformPrompt
from kuchikae.pipeline import KuchikaePipeline

TEMPLATES = {
    "自然に": "内容、数字、日時、固有名詞、否定条件は保ちつつ、言い回しを自然な日本語に変換してください。",
    "丁寧に": "次のテキストを「です・ます」調の丁寧な言葉遣いに変換してください。",
    "柔らかく": "次のテキストを柔らかく丁寧な表現に変換してください。",
    "短く": "次のテキストを簡潔に短く要約・変換してください。内容や固有名詞は変えないでください。",
    "カスタム": "",
}


def normalize_audio_path(audio_input):
    if audio_input is None:
        return None
    if isinstance(audio_input, tuple):
        sr, data = audio_input
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, data, sr)
            return tmp.name
    if isinstance(audio_input, dict):
        path = audio_input.get("orig_name") or audio_input.get("name") or audio_input.get("path")
        return str(path) if path else None
    path = getattr(audio_input, "path", None) or getattr(audio_input, "orig_name", None) or getattr(audio_input, "name", None)
    if path:
        return str(path)
    result = str(audio_input)
    return result if os.path.isfile(result) else None


def run_simple(
    pipeline: KuchikaePipeline,
    audio_input,
    live_streaming: bool = False,
) -> Generator:
    path = normalize_audio_path(audio_input)
    if path is None:
        yield gr.update(), "", "", ""
        return

    prompt = TextTransformPrompt(instruction=TEMPLATES["自然に"])
    stream_fn = pipeline.process_stream_live if live_streaming else pipeline.process_stream

    for status, src, txt, aud in stream_fn(path, prompt):
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


def run(
    audio_input,
    template_name: str,
    custom_prompt: str,
    pipeline: KuchikaePipeline,
    live_streaming: bool = False,
) -> Generator:
    path = normalize_audio_path(audio_input)
    if path is None:
        raise gr.Error("音声を録音またはアップロードしてください")

    if template_name == "カスタム" and custom_prompt.strip():
        prompt_text = custom_prompt
    elif template_name in TEMPLATES:
        prompt_text = TEMPLATES[template_name]
    else:
        prompt_text = TEMPLATES["自然に"]

    prompt = TextTransformPrompt(instruction=prompt_text)
    stream_fn = pipeline.process_stream_live if live_streaming else pipeline.process_stream

    for status, src, txt, aud in stream_fn(path, prompt):
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


def on_template_change(template_name: str):
    if template_name == "カスタム":
        return gr.update()
    text = TEMPLATES.get(template_name, "")
    return gr.update(value=text)
