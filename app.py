"""Kuchikae v0.1 — single-screen Gradio app."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.types import TextTransformPrompt, VoiceOutputPrompt


TEXT_TRANSFORM_DEFAULT = Path("prompts/text_transform_default.txt")
VOICE_OUTPUT_DEFAULT = Path("prompts/voice_output_default.txt")


def _load_prompt(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        return f.read().strip()


DEFAULT_TEXT_PROMPT = _load_prompt(TEXT_TRANSFORM_DEFAULT)
DEFAULT_VOICE_PROMPT = _load_prompt(VOICE_OUTPUT_DEFAULT)

pipeline = KuchikaePipeline()


def run(audio_path: str, text_prompt: str, voice_prompt: str):
    """Run the pipeline and return UI outputs."""
    if not audio_path or not os.path.isfile(str(audio_path)):
        raise gr.Error("Please upload or record an audio file first.")

    text_transform_prompt = TextTransformPrompt(instruction=text_prompt)
    voice_output_prompt = VoiceOutputPrompt(instruction=voice_prompt)

    result = pipeline.process(
        audio_path=str(audio_path),
        text_transform_prompt=text_transform_prompt,
        voice_output_prompt=voice_output_prompt,
    )

    vc_status = (
        f"Voice ready: {result.voice_ready}  |  "
        f"ID: {result.latency.stt_seconds:.3f}s STT, "
        f"{result.latency.text_transform_seconds:.3f}s transform, "
        f"{result.latency.voice_output_seconds:.3f}s output"
    )

    latency_lines = (
        f"STT: {result.latency.stt_seconds:.3f}s\n"
        f"Transform: {result.latency.text_transform_seconds:.3f}s\n"
        f"Voice output: {result.latency.voice_output_seconds:.3f}s\n"
        f"Total: {result.latency.total_seconds:.3f}s"
    )

    return (
        result.source_text,
        result.transformed_text,
        result.output_audio_path,
        vc_status,
        latency_lines,
    )


with gr.Blocks(title="Kuchikae v0.1") as demo:
    gr.Markdown("## Kuchikae — Speak once. Say it back your way.")

    with gr.Row():
        text_prompt = gr.Textbox(
            label="Text Transform Prompt",
            value=DEFAULT_TEXT_PROMPT,
            lines=3,
        )
        voice_prompt = gr.Textbox(
            label="Voice Output Prompt",
            value=DEFAULT_VOICE_PROMPT,
            lines=3,
        )

    audio_input = gr.Audio(label="Source Audio (upload)", type="filepath")
    submit_btn = gr.Button("Transform")

    with gr.Row():
        source_text = gr.Textbox(label="Source Transcript", lines=2)
        transformed_text = gr.Textbox(label="Transformed Text", lines=2)

    output_audio = gr.Audio(label="Output Audio", type="filepath")
    voice_status = gr.Textbox(label="Voice Context Status", lines=4)
    latency_report = gr.Textbox(label="Latency Report", lines=5)

    submit_btn.click(
        fn=run,
        inputs=[audio_input, text_prompt, voice_prompt],
        outputs=[source_text, transformed_text, output_audio, voice_status, latency_report],
    )


if __name__ == "__main__":
    demo.launch()
