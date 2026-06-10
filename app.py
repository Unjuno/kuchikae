"""Kuchikae v0.1 — push-to-talk Gradio app."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import soundfile as sf

from kuchikae.pipeline import create_pipeline
from kuchikae.types import TextTransformPrompt


TEXT_TRANSFORM_DEFAULT = Path("prompts/text_transform_default.txt")

DEFAULT_TEXT_PROMPT = TextTransformPrompt.from_file(TEXT_TRANSFORM_DEFAULT)


pipeline = create_pipeline()


CSS = """
#mic-btn {
    display: flex;
    justify-content: center;
    align-items: center;
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
    width: 80px !important;
    height: 80px !important;
    border-radius: 50% !important;
    border: none !important;
    background: var(--primary-600) !important;
    font-size: 0 !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    transition: transform 0.15s ease !important;
}
#mic-btn .record-button::before {
    display: none !important;
}
#mic-btn .record-button::after {
    content: "\U0001F3A4" !important;
    font-size: 32px !important;
    line-height: 1 !important;
}
#mic-btn .record-button:hover {
    transform: scale(1.08) !important;
    background: var(--primary-500) !important;
}
#mic-btn .record-button:active {
    transform: scale(0.94) !important;
}
#mic-btn .stop-button {
    width: 80px !important;
    height: 80px !important;
    border-radius: 50% !important;
    border: none !important;
    background: #ef4444 !important;
    font-size: 0 !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
}
#mic-btn .stop-button::before {
    display: none !important;
}
#mic-btn .stop-button::after {
    content: "\u25A0" !important;
    font-size: 28px !important;
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
"""


def _normalize_audio_path(audio_input):
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


def run(audio_input, text_prompt: str, use_defaults: bool):
    path = _normalize_audio_path(audio_input)
    if path is None:
        return None, gr.update(value=None)
    if use_defaults:
        text_prompt = DEFAULT_TEXT_PROMPT.instruction
    text_transform_prompt_obj = TextTransformPrompt(instruction=text_prompt)
    result = pipeline.process(
        audio_path=str(path),
        text_transform_prompt=text_transform_prompt_obj,
    )
    return result.output_audio_path, gr.update(value=None)


def on_use_defaults_change(use_defaults):
    if use_defaults:
        return gr.update(value=DEFAULT_TEXT_PROMPT.instruction, interactive=False)
    return gr.update(interactive=True)


with gr.Blocks(title="Kuchikae") as demo:
    gr.Markdown("## Kuchikae")

    audio_input = gr.Microphone(
        elem_id="mic-btn",
        show_label=False,
        buttons=[],
        waveform_options={"show_recording_waveform": False, "skip_length": 0},
        type="filepath",
        format="wav",
    )

    text_prompt = gr.Textbox(
        label="Text Transform Prompt",
        value=DEFAULT_TEXT_PROMPT.instruction,
        lines=2,
    )

    use_defaults = gr.Checkbox(
        label="Use default prompts",
        value=False,
    )

    output_audio = gr.Audio(label="Output Audio", type="filepath")

    use_defaults.change(
        on_use_defaults_change,
        inputs=[use_defaults],
        outputs=[text_prompt],
    )

    audio_input.stop_recording(
        run,
        inputs=[audio_input, text_prompt, use_defaults],
        outputs=[output_audio, audio_input],
    )


if __name__ == "__main__":
    demo.launch(css=CSS)
