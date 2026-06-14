"""Kuchikae UI — Gradio view layer (View Isolation).

Contains only the Gradio component tree (create_app).
CSS, JS, and handlers live in sibling modules.
"""

from __future__ import annotations

import logging

import gradio as gr
from gradio.themes.utils import colors

from kuchikae.domain.types import TextTransformPrompt, VoiceOutputPrompt
from kuchikae.pipeline import KuchikaePipeline
from kuchikae.ui.css import CSS
from kuchikae.ui.handlers import (
    TEMPLATES,
    _experimental_warning_html,
    on_template_change,
    on_template_category_change,
    run,
    run_simple,
)
from kuchikae.ui.templates import TEMPLATE_CATEGORIES
from kuchikae.ui.js import PTT_HTML, PTT_JS

logger = logging.getLogger("kuchikae.ui.app")


def _voice_analysis_pending_html() -> str:
    return '<span class="voice-analysis-label-inner">声の印象: 分析中...</span>'


_THEME = (
    gr.themes.Base(
        primary_hue=colors.purple,
        neutral_hue=colors.slate,
        text_size="lg",
        spacing_size="lg",
        radius_size="lg",
    ).set(
        body_background_fill="#18181B",
        body_background_fill_dark="#18181B",
        background_fill_primary="#18181B",
        background_fill_primary_dark="#18181B",
        background_fill_secondary="#252528",
        background_fill_secondary_dark="#252528",
        border_color_primary="#3F3F46",
        border_color_primary_dark="#3F3F46",
        body_text_color="#E4E4E7",
        body_text_color_dark="#E4E4E7",
        body_text_color_subdued="#A1A1AA",
        body_text_color_subdued_dark="#A1A1AA",
        button_primary_background_fill="#7C3AED",
        button_primary_background_fill_dark="#7C3AED",
        button_primary_background_fill_hover="#6D28D9",
        button_primary_background_fill_hover_dark="#6D28D9",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
    )
)


def create_app(
    pipeline: KuchikaePipeline,
    default_prompt: TextTransformPrompt,
    default_voice_prompt: VoiceOutputPrompt | None = None,
    live_streaming: bool = False,
) -> gr.Blocks:
    with gr.Blocks(title="Kuchikae", theme=_THEME) as demo:
        with gr.Column(elem_classes="wrap"):
            gr.HTML('<div id="title">Kuchikae</div>')
            stt_preset = gr.Radio(
                choices=[("速い", "fast"), ("バランス", "balanced"), ("高精度", "accurate")],
                value=getattr(pipeline, "stt_preset", "balanced"),
                label="音声認識モード",
                info="速度と精度のバランスを選びます。速いほど応答が早く、高精度ほど認識が正確です。",
                elem_id="stt-preset",
            )

            with gr.Tabs(elem_classes="tabs"):
                with gr.Tab("通常"):
                    template_category = gr.Radio(
                        elem_id="template-category",
                        label="カテゴリ",
                        choices=list(TEMPLATE_CATEGORIES.keys()),
                        value="標準",
                    )

                    template = gr.Dropdown(
                        elem_id="template-select",
                        label="どう言い換える？",
                        choices=TEMPLATE_CATEGORIES["標準"],
                        value="自然に",
                    )

                    voice_style = gr.Radio(
                        choices=["auto", "natural", "calm", "bright", "slow_clear"],
                        value="auto",
                        label="声の出し方",
                    )

                    voice_analysis_label = gr.HTML(
                        elem_id="voice-analysis-label",
                        value=_voice_analysis_pending_html(),
                    )

                    experimental_warning = gr.HTML(
                        elem_id="experimental-warning",
                        value=_experimental_warning_html("自然に"),
                    )

                    audio_input = gr.Audio(
                        elem_id="audio-input-wrap",
                        label="元の音声",
                        sources=["microphone", "upload"],
                        type="filepath",
                    )

                    run_btn = gr.Button("言い直す", elem_id="run-btn")

                    with gr.Row(elem_id="normal-text-compare"):
                        source_text = gr.Textbox(
                            elem_id="source-text",
                            label="聞き取った内容",
                            lines=4,
                            interactive=False,
                            placeholder="音声を録音すると、聞き取った内容が表示されます",
                        )
                        transformed_text = gr.Textbox(
                            elem_id="transformed-text",
                            label="言い直した内容",
                            lines=4,
                            interactive=False,
                            placeholder="言い換え結果が表示されます",
                        )

                    output_audio = gr.Audio(
                        elem_id="output-audio",
                        label="言い直し音声",
                        type="filepath",
                        autoplay=True,
                    )

                    status = gr.HTML(elem_id="status", value="", visible=True)

                    with gr.Accordion("詳細設定", open=False):
                        text_prompt = gr.Textbox(
                            elem_id="prompt-box",
                            label="LLMに渡す言い換え指示",
                            value=default_prompt.instruction,
                            lines=3,
                        )

                    def _run_handler(audio_value, template_value, text_prompt_value, stt_preset_value, voice_style_value):
                        # Disable button during processing
                        yield (gr.update(interactive=False, value="処理中..."),) + (gr.update(),) * 4
                        for result in run(
                            audio_value,
                            template_value,
                            text_prompt_value,
                            pipeline=pipeline,
                            live_streaming=live_streaming,
                            stt_preset=stt_preset_value,
                            voice_style=voice_style_value,
                        ):
                            yield (gr.update(interactive=True, value="言い直す"),) + result

                    template_category.change(
                        on_template_category_change,
                        inputs=[template_category],
                        outputs=[template],
                    )
                    template.change(
                        on_template_change,
                        inputs=[template],
                        outputs=[text_prompt, experimental_warning],
                    )
                    run_btn.click(
                        _run_handler,
                        inputs=[audio_input, template, text_prompt, stt_preset, voice_style],
                        outputs=[run_btn, output_audio, source_text, transformed_text, status, voice_analysis_label],
                    )

                with gr.Tab("簡易"):
                    simple_template_category = gr.Radio(
                        elem_id="simple-template-category",
                        label="カテゴリ",
                        choices=list(TEMPLATE_CATEGORIES.keys()),
                        value="標準",
                    )

                    simple_template = gr.Dropdown(
                        elem_id="simple-template-select",
                        label="どう言い換える？",
                        choices=TEMPLATE_CATEGORIES["標準"],
                        value="自然に",
                    )

                    simple_experimental_warning = gr.HTML(
                        elem_id="simple-experimental-warning",
                        value=_experimental_warning_html("自然に"),
                    )

                    gr.HTML(PTT_HTML, js_on_load=PTT_JS)

                    simple_audio = gr.Audio(
                        elem_id="simple-audio-wrap",
                        label="",
                        sources=["microphone"],
                        type="filepath",
                        visible=True,
                    )

                    simple_status = gr.HTML(elem_id="simple-status", value="", visible=True)

                    with gr.Row(elem_id="simple-text-compare"):
                        simple_source = gr.Textbox(
                            elem_id="simple-src",
                            label="聞き取った内容",
                            lines=3,
                            interactive=False,
                            placeholder="音声を録音すると表示されます",
                        )
                        simple_transformed = gr.Textbox(
                            elem_id="simple-trf",
                            label="言い直した内容",
                            lines=3,
                            interactive=False,
                            placeholder="言い換え結果が表示されます",
                        )

                    simple_output = gr.Audio(
                        elem_id="simple-output-audio",
                        label="言い直し音声",
                        type="filepath",
                        autoplay=True,
                    )

                    def _simple_template_change(name):
                        _, warning = on_template_change(name)
                        return warning

                    def _simple_category_change(category):
                        return on_template_category_change(category)

                    simple_template_category.change(
                        _simple_category_change,
                        inputs=[simple_template_category],
                        outputs=[simple_template],
                    )
                    simple_template.change(
                        _simple_template_change,
                        inputs=[simple_template],
                        outputs=[simple_experimental_warning],
                    )

                    def _run_simple_handler(audio_value, template_value, stt_preset_value):
                        yield from run_simple(
                            pipeline,
                            audio_value,
                            live_streaming=live_streaming,
                            stt_preset=stt_preset_value,
                            template_name=template_value,
                        )

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
