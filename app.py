"""Kuchikae v0.1 — single-screen Gradio app."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import soundfile as sf

from kuchikae.audio_cache import AudioCache
from kuchikae.pipeline import KuchikaePipeline, create_pipeline
from kuchikae.types import TextTransformPrompt, VoiceOutputPrompt


TEXT_TRANSFORM_DEFAULT = Path("prompts/text_transform_default.txt")
VOICE_OUTPUT_DEFAULT = Path("prompts/voice_output_default.txt")

# Presets for one-button mode: (text_prompt_file, voice_prompt_file)
PRESETS = {
    "カジュアル":  ("prompts/preset_casual.txt",       "prompts/preset_casual_voice.txt"),
    "丁寧な声で":   ("prompts/preset_polite.txt",        "prompts/preset_polite_voice.txt"),
    "感情豊か":     ("prompts/preset_emotional.txt",      "prompts/preset_emotional_voice.txt"),
}

# Pre-built default prompts (used when no file-based preset is selected)
DEFAULT_TEXT_PROMPT = TextTransformPrompt.from_file(TEXT_TRANSFORM_DEFAULT)
DEFAULT_VOICE_PROMPT = VoiceOutputPrompt.from_file(VOICE_OUTPUT_DEFAULT)


def _load_prompt(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        return f.read().strip()


pipeline = KuchikaePipeline()

# Shared audio cache — keeps recent utterances for multi-frame SE extraction.
audio_cache = AudioCache(max_references=5)


def _normalize_audio_path(audio_input):
    """Normalize Gradio's variable audio input into a real filesystem path.

    Gradio can pass: str (filepath), tuple (sr, data), dict (with "orig_name"), or None.
    Returns the resolved file path as a string.
    """
    if audio_input is None:
        raise gr.Error("No audio input received.")

    if isinstance(audio_input, tuple):
        sr, data = audio_input
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, data, sr)
            return tmp.name

    # dict form (e.g. from uploaded files in some Gradio versions).
    if isinstance(audio_input, dict):
        path = audio_input.get("orig_name") or audio_input.get("name") or audio_input.get("path")
        if not path:
            raise gr.Error(f"Could not resolve audio file path from dict: {audio_input!r}")
        return str(path)

    # string form (filepath).
    result = str(audio_input)
    if not os.path.isfile(result):
        raise gr.Error(f"No audio file found. (path={result!r})")
    return result


def _process_audio(audio_path, label, text_prompt, voice_prompt):
    """Process audio through the pipeline and return UI outputs."""
    if not os.path.isfile(str(audio_path)):
        raise gr.Error(f"{label}: No audio file found. (path={audio_path!r})")

    # Update reference cache.
    audio_cache.add_utterance(audio_path)

    text_transform_prompt_obj = TextTransformPrompt(instruction=text_prompt)
    voice_output_prompt_obj = VoiceOutputPrompt(
        instruction=voice_prompt,
        emotion=None,
    )

    result = pipeline.process(
        audio_path=str(audio_path),
        text_transform_prompt=text_transform_prompt_obj,
        voice_output_prompt=voice_output_prompt_obj,
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


def run(audio_input, text_prompt: str, voice_prompt: str):
    """Run the pipeline and return UI outputs."""
    path = _normalize_audio_path(audio_input)
    return _process_audio(path, "Manual", text_prompt, voice_prompt)


def run_one_button(audio_input):
    """Run the pipeline with pre-built default prompts."""
    path = _normalize_audio_path(audio_input)
    return _process_audio(
        path, "One Button", DEFAULT_TEXT_PROMPT.instruction, DEFAULT_VOICE_PROMPT.instruction
    )


def run_with_preset(audio_input: str, preset_name: str):
    """Run the pipeline with a pre-built prompt pair."""
    audio_path = _normalize_audio_path(audio_input)

    if not os.path.isfile(audio_path):
        raise gr.Error("Please upload an audio file first.")

    if preset_name not in PRESETS:
        raise gr.Error(f"Unknown preset: {preset_name}")

    if preset_name not in PRESETS:
        raise gr.Error(f"Unknown preset: {preset_name}")

    text_file, voice_file = PRESETS[preset_name]
    text_prompt_obj = TextTransformPrompt.from_file(text_file)
    voice_prompt_obj = VoiceOutputPrompt.from_file(voice_file)

    return _process_audio(audio_path, "Presets", text_prompt_obj.instruction, voice_prompt_obj.instruction)


with gr.Blocks(title="Kuchikae v0.1") as demo:
    gr.Markdown("## Kuchikae — Speak once. Say it back your way.")

    with gr.Tabs():
        # ── One-button mode (main UI) — upload first ────────────────
        with gr.Tab("One Button"):
            one_button_audio = gr.Audio(
                label="Record or upload audio",
                type="filepath",
                sources=["upload", "microphone"],
            )
            one_button_btn = gr.Button("▶ Transform.", size="lg")

            with gr.Row():
                ob_source_text = gr.Textbox(label="Source Text", lines=2)
                ob_transformed_text = gr.Textbox(label="Transformed Text", lines=2)

            ob_output_audio = gr.Audio(label="Output Audio (your voice)", type="filepath")
            ob_voice_status = gr.Textbox(label="Status", lines=4)
            ob_latency_report = gr.Textbox(label="Latency Report", lines=5)

            one_button_btn.click(
                fn=run_one_button,
                inputs=[one_button_audio],
                outputs=[ob_source_text, ob_transformed_text, ob_output_audio, ob_voice_status, ob_latency_report],
            )

        # ── Presets mode (sub-mode, upload-based) ─────────────────────
        with gr.Tab("Presets"):
            preset_selector = gr.Dropdown(
                label="Voice Style",
                choices=list(PRESETS.keys()),
                value="カジュアル",
            )
            presets_audio = gr.Audio(
                label="Source / Reference Audio",
                type="filepath",
            )
            submit_preset_btn = gr.Button("Transform with Preset")

            with gr.Row():
                ps_source_text = gr.Textbox(label="Source Text", lines=2)
                ps_transformed_text = gr.Textbox(label="Transformed Text", lines=2)

            ps_output_audio = gr.Audio(label="Output Audio", type="filepath")
            ps_voice_status = gr.Textbox(label="Status", lines=4)
            ps_latency_report = gr.Textbox(label="Latency Report", lines=5)

            submit_preset_btn.click(
                fn=run_with_preset,
                inputs=[presets_audio, preset_selector],
                outputs=[ps_source_text, ps_transformed_text, ps_output_audio, ps_voice_status, ps_latency_report],
            )

        # ── Manual mode (sub-mode, upload-based) ──────────────────────
        with gr.Tab("Manual"):
            text_prompt = gr.Textbox(
                label="Text Transform Prompt",
                value=DEFAULT_TEXT_PROMPT.instruction,
                lines=3,
            )
            voice_prompt = gr.Textbox(
                label="Voice Output Prompt",
                value=DEFAULT_VOICE_PROMPT.instruction,
                lines=3,
            )

            audio_input = gr.Audio(label="Source Audio (upload)", type="filepath")
            submit_btn = gr.Button("Transform")

            with gr.Row():
                source_text = gr.Textbox(label="Source Text", lines=2)
                transformed_text = gr.Textbox(label="Transformed Text", lines=2)

            output_audio = gr.Audio(label="Output Audio", type="filepath")
            voice_status = gr.Textbox(label="Status", lines=4)
            latency_report = gr.Textbox(label="Latency Report", lines=5)

            submit_btn.click(
                fn=run,
                inputs=[audio_input, text_prompt, voice_prompt],
                outputs=[source_text, transformed_text, output_audio, voice_status, latency_report],
            )


if __name__ == "__main__":
    demo.launch()
