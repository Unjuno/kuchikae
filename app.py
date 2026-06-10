"""Kuchikae v0.1 — push-to-talk Gradio app."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import soundfile as sf

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.types import TextTransformPrompt, VoiceOutputPrompt


TEXT_TRANSFORM_DEFAULT = Path("prompts/text_transform_default.txt")
VOICE_OUTPUT_DEFAULT = Path("prompts/voice_output_default.txt")

DEFAULT_TEXT_PROMPT = TextTransformPrompt.from_file(TEXT_TRANSFORM_DEFAULT)
DEFAULT_VOICE_PROMPT = VoiceOutputPrompt.from_file(VOICE_OUTPUT_DEFAULT)


pipeline = KuchikaePipeline()


def _normalize_audio_path(audio_input):
    if audio_input is None:
        return None
    if isinstance(audio_input, tuple):
        sr, data = audio_input
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, data, sr)
            return tmp.name
    path = getattr(audio_input, "path", None) or getattr(audio_input, "orig_name", None)
    if path:
        return str(path)
    result = str(audio_input)
    if not os.path.isfile(result):
        return None
    return result


def _process_audio(audio_path, text_prompt, voice_prompt):
    text_transform_prompt_obj = TextTransformPrompt(instruction=text_prompt)
    voice_output_prompt_obj = VoiceOutputPrompt(instruction=voice_prompt, emotion=None)
    result = pipeline.process(
        audio_path=str(audio_path),
        text_transform_prompt=text_transform_prompt_obj,
        voice_output_prompt=voice_output_prompt_obj,
    )
    return result.source_text, result.transformed_text, result.output_audio_path


def run(audio_input, text_prompt: str, voice_prompt: str, use_defaults: bool):
    path = _normalize_audio_path(audio_input)
    if path is None:
        return None, None, None
    if use_defaults:
        text_prompt = DEFAULT_TEXT_PROMPT.instruction
        voice_prompt = DEFAULT_VOICE_PROMPT.instruction
    return _process_audio(path, text_prompt, voice_prompt)


def on_use_defaults_change(use_defaults):
    if use_defaults:
        return (
            gr.update(value=DEFAULT_TEXT_PROMPT.instruction, interactive=False),
            gr.update(value=DEFAULT_VOICE_PROMPT.instruction, interactive=False),
        )
    return (
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


with gr.Blocks(title="Kuchikae v0.1") as demo:
    gr.Markdown("## Kuchikae")

    audio_input = gr.Microphone(
        label="Tap to record — tap again to stop & transform",
        type="filepath",
        format="wav",
    )

    text_prompt = gr.Textbox(
        label="Text Transform Prompt",
        value=DEFAULT_TEXT_PROMPT.instruction,
        lines=2,
    )
    voice_prompt = gr.Textbox(
        label="Voice Output Prompt",
        value=DEFAULT_VOICE_PROMPT.instruction,
        lines=2,
    )

    use_defaults = gr.Checkbox(
        label="Use default prompts",
        value=False,
    )

    source_text = gr.Textbox(label="Source Text (STT)", lines=2)
    transformed_text = gr.Textbox(label="Transformed Text", lines=2)
    output_audio = gr.Audio(label="Output Audio", type="filepath")

    use_defaults.change(
        on_use_defaults_change,
        inputs=[use_defaults],
        outputs=[text_prompt, voice_prompt],
    )

    audio_input.stop_recording(
        run,
        inputs=[audio_input, text_prompt, voice_prompt, use_defaults],
        outputs=[source_text, transformed_text, output_audio],
    )


if __name__ == "__main__":
    demo.launch()
