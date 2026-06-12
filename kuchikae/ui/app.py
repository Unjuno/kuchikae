"""Kuchikae UI — Gradio view layer (View Isolation).

Contains only the Gradio component tree (create_app).
CSS, JS, and handlers live in sibling modules.
"""

from __future__ import annotations

import logging

import gradio as gr

from kuchikae.domain.types import TextTransformPrompt, VoiceOutputPrompt
from kuchikae.pipeline import KuchikaePipeline
from kuchikae.ui.css import CSS
from kuchikae.ui.handlers import (
    TEMPLATES,
    on_template_change,
    run,
    run_simple,
)
from kuchikae.ui.js import PTT_HTML, PTT_JS

logger = logging.getLogger("kuchikae.ui.app")


def create_app(
    pipeline: KuchikaePipeline,
    default_prompt: TextTransformPrompt,
    default_voice_prompt: VoiceOutputPrompt | None = None,
    live_streaming: bool = False,
) -> gr.Blocks:
    with gr.Blocks(title="Kuchikae") as demo:
        with gr.Column(elem_classes="wrap"):
            gr.HTML('<div id="title">Kuchikae</div>')
            stt_preset = gr.Radio(
                choices=["fast", "balanced", "accurate"],
                value=getattr(pipeline, "stt_preset", "balanced"),
                label="音声認識モード",
                elem_id="stt-preset",
            )

            with gr.Tabs(elem_classes="tabs"):
                with gr.Tab("通常"):
                    template = gr.Dropdown(
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

                    voice_style = gr.Radio(
                        choices=["auto", "natural", "calm", "bright", "slow_clear", "custom"],
                        value="auto",
                        label="声の出し方",
                    )
                    with gr.Accordion("詳細設定", open=False):
                        voice_prompt = gr.Textbox(
                            elem_id="voice-prompt-box",
                            label="声の出し方プロンプト",
                            value=(default_voice_prompt.instruction if default_voice_prompt is not None else ""),
                            lines=3,
                        )

                    with gr.Row(elem_id="normal-text-compare"):
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

                    text_prompt = gr.Textbox(
                        elem_id="prompt-box",
                        label="LLMに渡す言い換え指示",
                        value=default_prompt.instruction,
                        lines=3,
                    )

                    def _run_handler(audio_value, template_value, text_prompt_value, voice_prompt_value, stt_preset_value, voice_style_value):
                        yield from run(
                            audio_value,
                            template_value,
                            text_prompt_value,
                            pipeline=pipeline,
                            live_streaming=live_streaming,
                            voice_output_prompt=voice_prompt_value,
                            stt_preset=stt_preset_value,
                            voice_style=voice_style_value,
                        )

                    template.change(
                        on_template_change,
                        inputs=[template],
                        outputs=[text_prompt],
                    )
                    run_btn.click(
                        _run_handler,
                        inputs=[audio_input, template, text_prompt, voice_prompt, stt_preset, voice_style],
                        outputs=[output_audio, source_text, transformed_text, status],
                    )

                with gr.Tab("簡易"):
                    simple_audio = gr.Audio(
                        elem_id="simple-audio-wrap",
                        label="",
                        sources=["microphone"],
                        type="filepath",
                    )

                    gr.HTML(PTT_HTML, js_on_load=PTT_JS)

                    simple_template = gr.Dropdown(
                        elem_id="simple-template-select",
                        label="どう言い換える？",
                        choices=list(TEMPLATES.keys()),
                        value="自然に",
                    )

                    with gr.Row(elem_id="simple-text-compare"):
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

                    def _run_simple_handler(audio_value, template_value, stt_preset_value):
                        yield from run_simple(
                            pipeline,
                            audio_value,
                            live_streaming=live_streaming,
                            stt_preset=stt_preset_value,
                            template_name=template_value,
                        )

                    simple_status = gr.HTML(elem_id="simple-status", value="", visible=True)

                    simple_audio.stop(
                        _run_simple_handler,
                        inputs=[simple_audio, simple_template, stt_preset],
                        outputs=[simple_output, simple_source, simple_transformed, simple_status],
                    )
                    simple_audio.change(
                        _run_simple_handler,
                        inputs=[simple_audio, simple_template, stt_preset],
                        outputs=[simple_output, simple_source, simple_transformed, simple_status],
                    )

    return demo
