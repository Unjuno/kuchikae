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
.gradio-container { max-width: 520px !important; margin: 0 auto !important; padding: 20px 16px !important; }
.wrap { background: #fff; border-radius: 16px; padding: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
#title { text-align: center; font-size: 22px; font-weight: 700; color: #1d1d1f; margin-bottom: 20px; }
#mic-wrap { display: flex; justify-content: center !important; margin-bottom: 16px; }
#mic-wrap .icon-button-wrapper { display: none !important; }
#mic-wrap label.container { border: none !important; box-shadow: none !important; padding: 0 !important; background: transparent !important; }
#mic-wrap .controls { justify-content: center !important; gap: 0 !important; }
#mic-wrap .record-button { width: 88px; height: 88px; border-radius: 50%; border: none; background: linear-gradient(135deg,#6e8efb,#a777e3); box-shadow: 0 6px 24px rgba(110,142,251,.3) !important; font-size: 0; transition: all .15s ease; }
#mic-wrap .record-button::before { display: none !important; }
#mic-wrap .record-button::after { content: "\U0001F3A4"; font-size: 32px; line-height: 1; }
#mic-wrap .record-button:hover { transform: scale(1.05); }
#mic-wrap .stop-button { width: 88px; height: 88px; border-radius: 50%; border: none; background: #ef4444; box-shadow: 0 6px 24px rgba(239,68,68,.3) !important; font-size: 0; transition: all .15s ease; }
#mic-wrap .stop-button::before { display: none !important; }
#mic-wrap .stop-button::after { content: "\u25A0"; font-size: 26px; line-height: 1; }
#mic-wrap .stop-button-paused, #mic-wrap .pause-button, #mic-wrap .resume-button, #mic-wrap .mic-select, #mic-wrap .timestamps, #mic-wrap .timestamp, #mic-wrap .microphone { display: none !important; }
#status { font-size: 11px; color: #9ca3af; text-align: center; min-height: 14px; }
#template-box { margin-bottom: 8px; }
#template-box label { font-size: 10px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }
#prompt-box textarea { border-radius: 8px; border: 1px solid #e5e7eb; padding: 8px 12px; font-size: 13px; color: #1d1d1f; background: #fafafa; min-height: 52px; }
#prompt-box textarea:focus { border-color: #6e8efb; background: #fff; }
#source-text label, #transformed-text label { display: none !important; }
#source-text textarea, #transformed-text textarea { border: none; background: transparent; padding: 4px 0; font-size: 13px; line-height: 1.5; resize: none; box-shadow: none; }
#source-text textarea { color: #9ca3af; font-style: italic; }
#transformed-text textarea { color: #1d1d1f; }
#output-audio { margin-top: 2px; }
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


def create_app(pipeline: KuchikaePipeline, default_prompt: TextTransformPrompt, live_streaming: bool = False) -> gr.Blocks:

    import logging
    logger = logging.getLogger("kuchikae.ui")

    def run(audio_input, text_prompt: str) -> Generator:
        path = normalize_audio_path(audio_input)
        if path is None:
            raise gr.Error("No audio")
        prompt = TextTransformPrompt(instruction=text_prompt)

        stream_fn = pipeline.process_stream_live if live_streaming else pipeline.process_stream
        for status, src, txt, aud in stream_fn(path, prompt):
            if status == "DONE":
                yield aud, src, txt, "DONE", None
            elif status == "VOX":
                yield gr.update(value=None), src, txt, "VOX", None
            elif status == "TXT":
                yield gr.update(value=None), src, "", "TXT", None
            elif status == "STT_PARTIAL":
                yield gr.update(value=None), src, "", "STT...", None
            else:
                yield gr.update(value=None), "", "", "STT", None

    def on_template_change(template_name: str):
        text = TEMPLATES.get(template_name, "")
        return gr.update(value=text)

    with gr.Blocks(title="Kuchikae") as demo:
        with gr.Column(elem_classes="wrap"):
            gr.HTML('<div id="title">Kuchikae</div>')

            audio_input = gr.Microphone(
                elem_id="mic-wrap",
                show_label=False,
                buttons=[],
                waveform_options={"show_recording_waveform": False, "skip_length": 0},
                type="filepath",
                format="wav",
            )

            status = gr.HTML(elem_id="status", value="", visible=True)

            with gr.Column(scale=0, min_width=0):
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
