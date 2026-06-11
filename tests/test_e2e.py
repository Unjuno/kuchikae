"""E2E tests for Kuchikae web UI.

Requires playwright (not installed by default):
    uv add --dev playwright
    playwright install chromium

Then run:
    uv run python -m pytest tests/test_e2e.py -v --browser chromium --headed
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


pytestmark = pytest.mark.skipif(
    not os.environ.get("KUCHIKAE_E2E"),
    reason="E2E tests require KUCHIKAE_E2E=1 and playwright installed",
)


@pytest.fixture(scope="module")
def app_url() -> str:
    url = os.environ.get("KUCHIKAE_URL", "http://127.0.0.1:7860")
    return url


@pytest.fixture(scope="module")
def dummy_wav() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, np.zeros(44100, dtype=np.float32), 44100)
    yield tmp.name
    os.unlink(tmp.name)


def test_app_loads(page, app_url):
    page.goto(app_url)
    assert "Kuchikae" in page.title()


def test_title_displayed(page, app_url):
    page.goto(app_url)
    title = page.locator("#title")
    assert title.is_visible()
    assert title.text_content() == "Kuchikae"


def test_two_tabs_exist(page, app_url):
    page.goto(app_url)
    tabs = page.locator("button[role='tab']")
    assert tabs.count() == 2
    assert tabs.nth(0).text_content() == "通常"
    assert tabs.nth(1).text_content() == "簡易"


def test_normal_tab_components(page, app_url):
    page.goto(app_url)
    assert page.locator("#template-select").is_visible()
    assert page.locator("#audio-input-wrap").is_visible()
    assert page.locator("#run-btn").is_visible()
    assert page.locator("#source-text").is_visible()
    assert page.locator("#transformed-text").is_visible()
    assert page.locator("#output-audio").is_visible()
    assert page.locator("#run-btn").text_content() == "言い直す"


def test_simple_tab_components(page, app_url):
    page.goto(app_url)
    page.locator("button[role='tab']:has-text('簡易')").click()
    page.wait_for_timeout(500)
    assert page.locator("#simple-audio-wrap").is_visible()
    assert page.locator("#simple-src").is_visible()
    assert page.locator("#simple-trf").is_visible()
    assert page.locator("#simple-output-audio").is_visible()
    assert page.locator("#simple-status").is_visible()


def test_simple_tab_no_run_button(page, app_url):
    page.goto(app_url)
    page.locator("button[role='tab']:has-text('簡易')").click()
    page.wait_for_timeout(500)
    assert page.locator("#run-btn").is_hidden() or not page.locator("#run-btn").is_visible()


def test_normal_tab_upload_file_and_run(page, app_url, dummy_wav):
    page.goto(app_url)
    file_input = page.locator("#audio-input-wrap input[type='file']")
    file_input.set_input_files(dummy_wav)
    page.wait_for_timeout(1000)
    page.locator("#run-btn").click()
    page.wait_for_timeout(2000)


def test_simple_tab_upload_does_not_exist(page, app_url):
    """Simple mode should not have an upload button (mic only)."""
    page.goto(app_url)
    page.locator("button[role='tab']:has-text('簡易')").click()
    page.wait_for_timeout(500)
    upload_btn = page.locator("#simple-audio-wrap button:has-text('Upload')")
    assert upload_btn.count() == 0


def test_no_contrast_issues(page, app_url):
    """Verify text-dark-on-dark-bg issues in both tabs."""
    page.goto(app_url)
    body = page.locator("body")
    bg = body.evaluate("el => getComputedStyle(el).backgroundColor")
    color = body.evaluate("el => getComputedStyle(el).color")
    assert bg == "rgb(24, 24, 27)"
    assert color == "rgb(228, 228, 231)"

    card = page.locator(".main > .wrap")
    card_bg = card.evaluate("el => getComputedStyle(el).backgroundColor")
    card_color = card.evaluate("el => getComputedStyle(el).color")
    assert card_bg == "rgb(37, 37, 40)"
    assert card_color == "rgb(228, 228, 231)"
