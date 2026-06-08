"""Text transformation backend interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from kuchikae.types import TextTransformPrompt


class TextTransformBackend(ABC):
    """Abstract base for prompt-conditioned text transformation backends."""

    @abstractmethod
    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        """Transform source text according to a free-form prompt."""
        raise NotImplementedError


class DummyTextTransformBackend(TextTransformBackend):
    """Deterministic dummy text transformer for v0.1 scaffold tests."""

    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        """Return a non-empty transformed string while preserving prompt shape."""
        instruction = prompt.instruction.strip()
        if not instruction:
            return f"[transformed] {text}"
        return f"[transformed according to prompt] {text}"
