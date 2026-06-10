"""Kuchikae UI — Gradio view layer (View Isolation)."""

from __future__ import annotations

import os
import tempfile

import gradio as gr
import soundfile as sf

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.types import TextTransformPrompt

CSS = """
body {
    background: #f5f5f7 !important;
}
.gradio-container {
    max-width: 520px !important;
    margin: 40px auto !important;
}
#title {
    text-align: center;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #1d1d1f !important;
    margin-bottom: 20px !important;
    letter-spacing: -0.02em !important;
}
#mic-btn {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    margin-bottom: 20px !important;
}
#mic-btn .icon-button-wrapper {
    display: none !important;
}
#mic-btn .component-wrapper {
    padding: 0 !important;
}
#mic-btn .controls {
    justify-content: center !important;
}
#mic-btn .wrapper {
    justify-content: center;
}
#mic-btn .record-button {
    width: 120px !important;
    height: 120px !important;
    border-radius: 50% !important;
    border: none !important;
    background: linear-gradient(135deg, #6e8efb, #a777e3) !important;
    box-shadow: 0 8px 32px rgba(110, 142, 251, 0.35) !important;
    font-size: 0 !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
#mic-btn .record-button::before {
    display: none !important;
}
#mic-btn .record-button::after {
    content: "\U0001F3A4" !important;
    font-size: 44px !important;
    line-height: 1 !important;
}
#mic-btn .record-button:hover {
    transform: scale(1.06) !important;
    box-shadow: 0 12px 40px rgba(110, 142, 251, 0.45) !important;
}
#mic-btn .record-button:active {
    transform: scale(0.96) !important;
}
#mic-btn .stop-button {
    width: 120px !important;
    height: 120px !important;
    border-radius: 50% !important;
    border: none !important;
    background: #ef4444 !important;
    box-shadow: 0 8px 32px rgba(239, 68, 68, 0.35) !important;
    font-size: 0 !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    transition: all 0.2s ease !important;
}
#mic-btn .stop-button::before {
    display: none !important;
}
#mic-btn .stop-button::after {
    content: "\u25A0" !important;
    font-size: 36px !important;
    line-height: 1 !important;
}
#mic-btn .stop-button-paused,
#mic-btn .pause-button,
#mic-btn .resume-button,
#mic-btn .mic-select,
#mic-btn .timestamps,
#mic-btn .timestamp,
#mic-btn .microphone {
    display: none !important;
}
#template-box label,
#prompt-box label {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #9ca3af !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
#prompt-box textarea {
    border-radius: 10px !important;
    border: 1.5px solid #e5e7eb !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    color: #1d1d1f !important;
    background: white !important;
    transition: border-color 0.15s ease !important;
    resize: vertical !important;
    min-height: 60px !important;
}
#prompt-box textarea:focus {
    border-color: #6e8efb !important;
    box-shadow: 0 0 0 3px rgba(110, 142, 251, 0.15) !important;
}
#source-text label,
#transformed-text label {
    display: none !important;
}
#source-text textarea,
#transformed-text textarea {
    border: none !important;
    background: transparent !important;
    padding: 2px 0 !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
    resize: none !important;
    box-shadow: none !important;
    min-height: 20px !important;
}
#source-text textarea {
    color: #6b7280 !important;
    font-style: italic !important;
}
#transformed-text textarea {
    color: #1d1d1f !important;
    font-weight: 500 !important;
}
#output-audio {
    margin-top: 8px !important;
}
#output-audio audio {
    height: 40px !important;
    border-radius: 6px !important;
}
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

    def run(audio_input, text_prompt: str):
        path = normalize_audio_path(audio_input)
        if path is None:
            raise gr.Error("No audio captured.")
        prompt = TextTransformPrompt(instruction=text_prompt)
        result = pipeline.process(
            audio_path=str(path),
            text_transform_prompt=prompt,
        )
        return (result.output_audio_path, result.source_text, result.transformed_text,
                gr.update(value=None))

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
            outputs=[output_audio, source_text, transformed_text, audio_input],
        )

    return demo
