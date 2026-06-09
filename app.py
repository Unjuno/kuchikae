"""Kuchikae v0.1 — single-screen Gradio app."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

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


def run(audio_path: str, text_prompt: str, voice_prompt: str):
    """Run the pipeline and return UI outputs."""
    if not audio_path or not os.path.isfile(str(audio_path)):
        raise gr.Error("Please upload or record an audio file first.")

    # Update reference cache.
    audio_cache.add_utterance(audio_path)

    text_transform_prompt = TextTransformPrompt(instruction=text_prompt)
    voice_output_prompt = VoiceOutputPrompt(
        instruction=voice_prompt,
        emotion=None,  # Can be extended to auto-detect from prompt content.
    )

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


def run_one_button(audio_path: str):
    """Run the pipeline with pre-built default prompts."""
    if not audio_path or not os.path.isfile(str(audio_path)):
        raise gr.Error("Please upload or record an audio file first.")

    # Update reference cache.
    audio_cache.add_utterance(audio_path)

    result = pipeline.process(
        audio_path=str(audio_path),
        text_transform_prompt=DEFAULT_TEXT_PROMPT,
        voice_output_prompt=DEFAULT_VOICE_PROMPT,
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


def run_with_preset(audio_path: str, preset_name: str):
    """Run the pipeline with a pre-built prompt pair."""
    if not audio_path or not os.path.isfile(str(audio_path)):
        raise gr.Error("Please upload or record an audio file first.")

    if preset_name not in PRESETS:
        raise gr.Error(f"Unknown preset: {preset_name}")

    text_file, voice_file = PRESETS[preset_name]
    text_prompt = TextTransformPrompt.from_file(text_file)
    voice_prompt = VoiceOutputPrompt.from_file(voice_file)

    # Update reference cache.
    audio_cache.add_utterance(audio_path)

    result = pipeline.process(
        audio_path=str(audio_path),
        text_transform_prompt=text_prompt,
        voice_output_prompt=voice_prompt,
    )

    vc_status = (
        f"Preset: {preset_name}  |  "
        f"Voice ready: {result.voice_ready}"
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

    with gr.Tabs():
        # ── Manual mode (2 prompts) ─────────────────────────────
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

            audio_input = gr.Audio(label="Source Audio", type="filepath", sources=["upload"])
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

        # ── One-button mode ───────────────────────────────────────
        with gr.Tab("One Button"):
            one_button_audio = gr.Audio(label="Audio (upload)", type="filepath", sources=["upload"])
            one_button_btn = gr.Button("▶ Play back in your voice!", size="lg")

            with gr.Row():
                ob_source_text = gr.Textbox(label="Source Transcript", lines=2)
                ob_transformed_text = gr.Textbox(label="Transformed Text", lines=2)

            ob_output_audio = gr.Audio(label="Output Audio", type="filepath")
            ob_voice_status = gr.Textbox(label="Status", lines=4)
            ob_latency_report = gr.Textbox(label="Latency Report", lines=5)

            one_button_btn.click(
                fn=run_one_button,
                inputs=[one_button_audio],
                outputs=[ob_source_text, ob_transformed_text, ob_output_audio, ob_voice_status, ob_latency_report],
            )

        # ── Presets mode ──────────────────────────────────────────
        with gr.Tab("Presets"):
            preset_selector = gr.Dropdown(
                label="Voice Style",
                choices=list(PRESETS.keys()),
                value="カジュアル",
            )
            presets_audio = gr.Audio(label="Source Audio", type="filepath", sources=["upload"])
            submit_preset_btn = gr.Button("Transform with Preset")

            with gr.Row():
                ps_source_text = gr.Textbox(label="Source Transcript", lines=2)
                ps_transformed_text = gr.Textbox(label="Transformed Text", lines=2)

            ps_output_audio = gr.Audio(label="Output Audio", type="filepath")
            ps_voice_status = gr.Textbox(label="Status", lines=4)
            ps_latency_report = gr.Textbox(label="Latency Report", lines=5)

            submit_preset_btn.click(
                fn=run_with_preset,
                inputs=[presets_audio, preset_selector],
                outputs=[ps_source_text, ps_transformed_text, ps_output_audio, ps_voice_status, ps_latency_report],
            )


if __name__ == "__main__":
    demo.launch()
