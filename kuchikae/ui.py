"""Kuchikae UI — Gradio view layer (View Isolation)."""

from __future__ import annotations

import os
import tempfile
from typing import Generator

import gradio as gr
import soundfile as sf

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.types import TextTransformPrompt

CSS = """
body { background: #f5f5f7 !important; }
.gradio-container { max-width: 520px !important; margin: 40px auto !important; }
#title { text-align: center; font-size: 28px; font-weight: 700; color: #1d1d1f; margin-bottom: 20px; letter-spacing: -0.02em; }
#mic-btn { display: flex; justify-content: center; align-items: center; margin-bottom: 20px; }
#mic-btn .icon-button-wrapper { display: none !important; }
#mic-btn .component-wrapper { padding: 0 !important; }
#mic-btn .controls { justify-content: center !important; }
#mic-btn .wrapper { justify-content: center; }
#mic-btn .record-button { width: 120px; height: 120px; border-radius: 50%; border: none; background: linear-gradient(135deg,#6e8efb,#a777e3); box-shadow: 0 8px 32px rgba(110,142,251,.35); font-size: 0; align-items: center; justify-content: center; margin: 0; transition: all .2s ease; cursor: pointer; }
#mic-btn .record-button::before { display: none !important; }
#mic-btn .record-button::after { content: "\U0001F3A4"; font-size: 44px; line-height: 1; }
#mic-btn .record-button:hover { transform: scale(1.06); box-shadow: 0 12px 40px rgba(110,142,251,.45); }
#mic-btn .record-button:active { transform: scale(.96); }
#mic-btn .stop-button { width: 120px; height: 120px; border-radius: 50%; border: none; background: #ef4444; box-shadow: 0 8px 32px rgba(239,68,68,.35); font-size: 0; align-items: center; justify-content: center; margin: 0; transition: all .2s ease; }
#mic-btn .stop-button::before { display: none !important; }
#mic-btn .stop-button::after { content: "\u25A0"; font-size: 36px; line-height: 1; }
#mic-btn .stop-button-paused, #mic-btn .pause-button, #mic-btn .resume-button, #mic-btn .mic-select, #mic-btn .timestamps, #mic-btn .timestamp, #mic-btn .microphone { display: none !important; }
#status { font-size: 11px; color: #9ca3af; text-align: center; margin-bottom: 8px; letter-spacing: .08em; min-height: 16px; }
#template-box label, #prompt-box label { font-size: 11px; font-weight: 600; color: #9ca3af; letter-spacing: .06em; text-transform: uppercase; }
#prompt-box textarea { border-radius: 10px; border: 1.5px solid #e5e7eb; padding: 10px 14px; font-size: 13px; line-height: 1.5; color: #1d1d1f; background: white; transition: border-color .15s ease; resize: vertical; min-height: 60px; }
#prompt-box textarea:focus { border-color: #6e8efb; box-shadow: 0 0 0 3px rgba(110,142,251,.15); }
#source-text label, #transformed-text label { display: none !important; }
#source-text textarea, #transformed-text textarea { border: none; background: transparent; padding: 2px 0; font-size: 12px; line-height: 1.5; resize: none; box-shadow: none; min-height: 20px; }
#source-text textarea { color: #6b7280; font-style: italic; }
#transformed-text textarea { color: #1d1d1f; font-weight: 500; }
#output-audio { margin-top: 8px; }
#output-audio audio { height: 40px; border-radius: 6px; }
"""

TEMPLATES = {
    "Custom": "",
    "Polite": "次のテキストを「です・ます」調の丁寧な言葉遣いに変換してください。",
    "Casual": "次のテキストを友達同士のカジュアルな口調（タメ口・普通体）に変換してください。",
    "Summarize": "次のテキストを簡潔に1〜2文に要約してください。内容や固有名詞は変えないでください。",
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


def create_app(pipeline: KuchikaePipeline, default_prompt: TextTransformPrompt) -> gr.Blocks:

    import logging
    logger = logging.getLogger("kuchikae.ui")

    def run(audio_input, text_prompt: str) -> Generator:
        path = normalize_audio_path(audio_input)
        if path is None:
            raise gr.Error("No audio")
        prompt = TextTransformPrompt(instruction=text_prompt)

        for status, src, txt, aud in pipeline.process_stream(path, prompt):
            if status == "DONE":
                yield aud, src, txt, "DONE", None
            elif status == "VOX":
                yield gr.update(value=None), src, txt, "VOX", None
            elif status == "TXT":
                yield gr.update(value=None), src, "", "TXT", None
            else:
                yield gr.update(value=None), "", "", "STT", None

    def on_template_change(template_name: str):
        text = TEMPLATES.get(template_name, "")
        return gr.update(value=text)

    with gr.Blocks(title="Kuchikae") as demo:
        gr.HTML('<div id="title">Kuchikae</div>')

        audio_input = gr.Microphone(
            elem_id="mic-btn",
            show_label=False,
            buttons=[],
            waveform_options={"show_recording_waveform": False, "skip_length": 0},
            type="filepath",
            format="wav",
        )

        status = gr.HTML(elem_id="status", value="", visible=True)

        template = gr.Radio(
            elem_id="template-box",
            label="TEMPLATE",
            choices=list(TEMPLATES.keys()),
            value="Custom",
        )

        text_prompt = gr.Textbox(
            elem_id="prompt-box",
            show_label=False,
            value=default_prompt.instruction,
            lines=2,
        )

        source_text = gr.Textbox(
            elem_id="source-text",
            show_label=False,
            lines=1,
            interactive=False,
            visible=True,
        )

        transformed_text = gr.Textbox(
            elem_id="transformed-text",
            show_label=False,
            lines=1,
            interactive=False,
            visible=True,
        )

        output_audio = gr.Audio(
            elem_id="output-audio",
            show_label=False,
            type="filepath",
            autoplay=True,
        )

        template.change(on_template_change, inputs=[template], outputs=[text_prompt])

        audio_input.stop_recording(
            run,
            inputs=[audio_input, text_prompt],
            outputs=[output_audio, source_text, transformed_text, status, audio_input],
        )

    return demo
