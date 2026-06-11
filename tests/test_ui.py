"""Tests for UI structure (create_app)."""

from __future__ import annotations

import gradio as gr

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.domain.types import TextTransformPrompt
from kuchikae.ui import CSS, TEMPLATES, create_app


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


def test_create_app_ptt_button():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    assert "ptt-container" in config_str
    assert "ptt-btn" in config_str
    assert "pttStart" in config_str
    assert "pttStop" in config_str
    assert "押して話す" in config_str


def test_create_app_templates_in_config():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt)
    config_str = str(demo.config)
    for name in TEMPLATES:
        assert name in config_str


def test_create_app_live_streaming_param():
    pipeline = KuchikaePipeline()
    prompt = TextTransformPrompt(instruction="test")
    demo = create_app(pipeline, prompt, live_streaming=True)
    assert isinstance(demo, gr.Blocks)
