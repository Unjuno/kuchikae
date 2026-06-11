"""Kuchikae UI — view layer (View Isolation)."""

from kuchikae.ui.app import create_app
from kuchikae.ui.css import CSS
from kuchikae.ui.handlers import TEMPLATES, normalize_audio_path, run_simple

__all__ = [
    "CSS",
    "TEMPLATES",
    "create_app",
    "normalize_audio_path",
    "run_simple",
]
