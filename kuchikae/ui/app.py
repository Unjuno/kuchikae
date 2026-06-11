"""Kuchikae UI — Gradio view layer (View Isolation).

Contains only the Gradio component tree (create_app).
CSS, JS, and handlers live in sibling modules.
"""

from __future__ import annotations

import functools
import logging

import gradio as gr

from kuchikae.domain.types import TextTransformPrompt
from kuchikae.pipeline import KuchikaePipeline
from kuchikae.ui.css import CSS
from kuchikae.ui.handlers import (
    TEMPLATES,
    on_template_change,
    run,
    run_simple,
)

logger = logging.getLogger("kuchikae.ui.app")


def create_app(
    pipeline: KuchikaePipeline,
    default_prompt: TextTransformPrompt,
    live_streaming: bool = False,
) -> gr.Blocks:
    with gr.Blocks(title="Kuchikae") as demo:
        with gr.Column(elem_classes="wrap"):
            gr.HTML('<div id="title">Kuchikae</div>')

            with gr.Tabs(elem_classes="tabs"):
                with gr.Tab("通常"):
                    template = gr.Radio(
                        elem_id="template-select",
                        label="どう言い換える？",
                        choices=list(TEMPLATES.keys()),
                        value="自然に",
                    )

                    audio_input = gr.Audio(
                        elem_id="audio-input-wrap",
                        label="元の音声",
                        sources=["microphone", "upload"],
                        type="filepath",
                    )

                    run_btn = gr.Button("言い直す", elem_id="run-btn")

                    with gr.Row(elem_id="text-compare"):
                        source_text = gr.Textbox(
                            elem_id="source-text",
                            label="聞き取った内容",
                            lines=4,
                            interactive=False,
                        )
                        transformed_text = gr.Textbox(
                            elem_id="transformed-text",
                            label="言い直した内容",
                            lines=4,
                            interactive=False,
                        )

                    output_audio = gr.Audio(
                        elem_id="output-audio",
                        label="言い直し音声",
                        type="filepath",
                        autoplay=True,
                    )

                    status = gr.HTML(elem_id="status", value="", visible=True)

                    with gr.Accordion("詳細プロンプト", open=False):
                        text_prompt = gr.Textbox(
                            elem_id="prompt-box",
                            label="LLMに渡す言い換え指示",
                            value=default_prompt.instruction,
                            lines=3,
                        )

                    template.change(
                        on_template_change,
                        inputs=[template],
                        outputs=[text_prompt],
                    )
                    run_btn.click(
                        functools.partial(run, pipeline=pipeline, live_streaming=live_streaming),
                        inputs=[audio_input, template, text_prompt],
                        outputs=[output_audio, source_text, transformed_text, status],
                    )

                with gr.Tab("簡易"):
                    simple_audio = gr.Audio(
                        elem_id="simple-audio",
                        label="音声を録音",
                        sources=["microphone"],
                        type="filepath",
                    )

                    with gr.Row(elem_id="text-compare"):
                        simple_source = gr.Textbox(
                            elem_id="simple-src",
                            label="聞き取った内容",
                            lines=4,
                            interactive=False,
                        )
                        simple_transformed = gr.Textbox(
                            elem_id="simple-trf",
                            label="言い直した内容",
                            lines=4,
                            interactive=False,
                        )

                    simple_output = gr.Audio(
                        elem_id="simple-output-audio",
                        label="言い直し音声",
                        type="filepath",
                        autoplay=True,
                    )

                    simple_status = gr.HTML(elem_id="simple-status", value="", visible=True)

                    simple_audio.stop(
                        functools.partial(run_simple, pipeline=pipeline, live_streaming=live_streaming),
                        inputs=[simple_audio],
                        outputs=[simple_output, simple_source, simple_transformed, simple_status],
                    )

    return demo
