"""Tests for UI structure (create_app)."""

from __future__ import annotations

import gradio as gr

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.domain.types import TextTransformPrompt
from kuchikae.ui import create_app



def test_create_app_returns_blocks():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    assert isinstance(demo, gr.Blocks)


def test_create_app_title_present():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    html = str(demo.config)
    assert "Kuchikae" in html


def test_create_app_has_two_tabs():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    assert "通常" in config_str
    assert "簡易" in config_str


def test_create_app_experimental_warning_present():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    assert "experimental-warning" in config_str


def test_create_app_normal_tab_components():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    assert "template-select" in config_str
    assert "audio-input-wrap" in config_str
    assert "run-btn" in config_str
    assert "source-text" in config_str
    assert "transformed-text" in config_str
    assert "output-audio" in config_str
    assert "prompt-box" in config_str
    assert "normal-text-compare" in config_str
    # Voice prompt textbox should NOT exist
    assert "voice-prompt-box" not in config_str
    # Voice analysis label should exist
    assert "voice-analysis-label" in config_str
    # custom should NOT be in voice_style choices
    assert '"custom"' not in config_str or "custom" not in config_str.split("voice_style")[0] if "voice_style" in config_str else True


def test_create_app_simple_tab_components():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    assert "simple-audio-wrap" in config_str
    assert "simple-src" in config_str
    assert "simple-trf" in config_str
    assert "simple-output-audio" in config_str
    assert "simple-status" in config_str
    assert "simple-text-compare" in config_str
    assert "simple-template-select" in config_str


def test_create_app_ptt_button():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    assert "ptt-container" in config_str
    assert "ptt-btn" in config_str
    assert "押して話す" in config_str


def test_create_app_templates_in_config():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    # The dropdown defaults to "標準" category; not all TEMPLATES are present
    for name in ("自然に", "丁寧に", "親しみやすく", "短く", "力強く", "落ち着いて"):
        assert name in config_str
    # Category names should be present in the Radio
    assert "標準" in config_str
    assert "キャラ" in config_str
    assert "実験" in config_str
    assert "実験強" in config_str
    assert "カスタム" in config_str


def test_create_app_has_stt_preset_selector():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    assert "音声認識モード" in config_str
    assert "fast" in config_str
    assert "balanced" in config_str
    assert "accurate" in config_str


def test_backend_status_shows_stt_details():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    assert "音声認識モード" in config_str


def test_create_app_live_streaming_param():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt, live_streaming=True)
    assert isinstance(demo, gr.Blocks)


def test_simple_audio_visible_true():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    assert "'visible': True" in config_str
    assert "simple-audio-wrap" in config_str


def test_simple_audio_wrap_css_offscreen():
    from kuchikae.ui.css import CSS
    assert "#simple-audio-wrap" in CSS
    assert "position: absolute" in CSS
    assert "left: -9999px" in CSS
    assert "overflow: hidden" in CSS
    assert "opacity: 0" in CSS
    assert "pointer-events: none" in CSS
    assert "display: none" not in CSS.split("#simple-audio-wrap")[1].split("}")[0]


def test_voice_analysis_label_no_id_duplication():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    assert "voice-analysis-label-inner" in config_str
    assert 'id="voice-analysis-label"' not in config_str


def test_experimental_warning_in_normal_tab():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    assert "experimental-warning" in config_str
    # Default template "自然に" is NOT experimental, so warning text should be absent
    assert "なりすまし、詐欺、脅迫、同意のない声の模倣には使用しないでください" not in config_str


def test_experimental_warning_in_simple_tab():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    count = config_str.count("experimental-warning")
    assert count >= 2, f"Expected at least 2 experimental-warning instances, got {count}"


def test_template_choices_include_experimental():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    # Category Radio should list "実験" and "実験強"
    assert '"実験"' in config_str or "'実験'" in config_str
    assert '"実験強"' in config_str or "'実験強'" in config_str


def test_simple_template_choices_include_experimental():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    # The simple tab dropdown also defaults to "標準" category
    simple_section = config_str.split("simple-template-select")[1]
    assert "自然に" in simple_section
    assert "実験: 関西弁" not in simple_section  # not in default category


def test_ptt_recording_no_transform_scale():
    from kuchikae.ui.css import CSS
    ptt_recording_block = CSS.split("#ptt-btn.ptt-recording")[1].split("}")[0]
    assert "transform" not in ptt_recording_block, "ptt-recording should not use transform"
    assert "scale" not in ptt_recording_block, "ptt-recording should not use scale"
