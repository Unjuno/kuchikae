"""TextTransformBackend and DummyTextTransformBackend."""

from __future__ import annotations


class TextTransformBackend:
    """Abstract base for text transformation backends."""

    def transform(  # pragma: no cover
        self,
        text: str,
        prompt,
    ) -> str:
        raise NotImplementedError


class DummyTextTransformBackend(TextTransformBackend):
    """Accepts a TextTransformPrompt and returns a non-empty transformed string.

    Does not implement fixed style-dropdown logic; the prompt is free-form text.
    """

    def transform(  # type: ignore[override]
        self,
        text: str,
        prompt,
    ) -> str:
        if hasattr(prompt, "prompt_text"):
            return f"[transformed] {text}"
        return f"[transformed] {text}"
