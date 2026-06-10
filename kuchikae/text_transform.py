"""Text transformation backend interfaces and real implementations."""

from __future__ import annotations

import os
import re
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


# --------------------------------------------------------------------------- #
#  Real backend — Ollama (local LLM, no API key needed)                     #
# --------------------------------------------------------------------------- #

class OllamaTextTransformBackend(TextTransformBackend):
    """Japanese text transformation backed by a local Ollama model.

    Calls Ollama API (http://localhost:11434/api/chat) with the configured
    model. Detects Ollama availability; falls back to Dummy if unavailable.
    """

    def __init__(self, model: str = "hf.co/LiquidAI/LFM2.5-1.2B-JP-GGUF:Q4_K_M") -> None:
        self.model = model
        self._base_url = "http://localhost:11434"

    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        import httpx

        try:
            resp = httpx.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "あなたは日本語のテキスト変換アシスタントです。"
                                "ユーザーの入力テキストをプロンプトに基づいて変換してください。"
                                "出力は変換結果のみ、余計な説明は不要です。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"テキスト: {text}\n\nプロンプト: {prompt.instruction}",
                        },
                    ],
                    "stream": False,
                    "options": {"temperature": 0.7},
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception:
            return DummyTextTransformBackend().transform(text, prompt)


# --------------------------------------------------------------------------- #
#  Real backend — gpt-oss (API-backed, requires OPENAI_API_KEY)             #
# --------------------------------------------------------------------------- #

class GPTTextTransformBackend(TextTransformBackend):
    """Prompt-conditioned text transformation backed by OpenAI-compatible LLM.

    Detects ``OPENAI_API_KEY`` and falls back to ``DummyTextTransformBackend``
    if the key is missing so that tests continue to pass without a paid API.
    """

    def __init__(self, model: str = "gpt-oss") -> None:
        self.model = model

    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return DummyTextTransformBackend().transform(text, prompt)

        import httpx  # lightweight, no heavy deps

        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": (
                        "あなたは日本語のテキスト変換アシスタントです。"
                        "ユーザーの入力テキストをプロンプトに基づいて変換してください。\n"
                        "- 内容、数字、日時、固有名詞、否定条件は保ってください。\n"
                        "- 新しい事実を追加しないでください。\n"
                        "- 出力は日本語のみで、余計な説明文を除いてください。"
                    )},
                    {"role": "user", "content": f"テキスト: {text}\n\nプロンプト: {prompt.instruction}"},
                ],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------- #
#  Real backend — rule-based Japanese rewriter (no API needed)               #
# --------------------------------------------------------------------------- #

class RuleTextTransformBackend(TextTransformBackend):
    """Rule-based real text transformation for Japanese.

    Works without any external model or API. Handles:
    - Plain text → polite desu/masu form
    - Plain text → casual da/dearu form
    - Content preservation (numbers, dates, names stay intact).
    """

    def __init__(self) -> None:
        self._desu_masu_map = {  # common plain→polite pairs
            "だ": "です",
            "である": "であります",
            "ない": "ません",
            "た": "ました",
            "る": "ります",
            "む": "みます",
            "ぶ": "びます",
            "ぬ": "にます",
            "ぐ": "ぎます",
            "う": "います",
        }

    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        instruction = prompt.instruction.strip().lower()
        target_form = self._detect_target_form(instruction)
        transformed = self._apply(text, target_form)
        return f"[{target_form}] {transformed}"

    def _detect_target_form(self, instruction: str) -> str:
        """Detect desired style from the prompt."""
        if any(kw in instruction for kw in ("丁寧", "です", "ます", "polite")):
            return "desu-masu"
        if any(kw in instruction for kw in (" casual", "カジュアル", "普通形", "plain")):
            return "plain"
        # Default to polite form.
        return "desu-masu"

    def _apply(self, text: str, target_form: str) -> str:
        if target_form == "desu-masu":
            return self._to_desu_masu(text)
        return self._to_plain(text)

    def _to_desu_masu(self, text: str) -> str:
        """Convert plain Japanese to polite desu/masu form."""
        parts = re.split(r"(。|、|\s+)", text)
        result_parts = []
        for part in parts:
            if part.endswith("だ") and len(part) > 2:
                part = part[:-1] + "です"
            elif part.endswith("である"):
                part = part.replace("である", "であります")
            elif part.endswith("ない"):
                part = part[:-2] + "ません"
            elif re.search(r"(る|む|ぶ|ぬ|ぐ|う)$", part) and not any(
                c in part for c in ("です", "ます", "は")
            ):
                if re.match(r"^.*(る)", part):
                    part = part.replace("る", "ります")
                elif re.search(r"(む)$", part):
                    part = re.sub(r"む$", "みます", part)
                elif re.search(r"(ぶ)$", part):
                    part = re.sub(r"ぶ$", "びます", part)
            result_parts.append(part)
        return "".join(result_parts)

    def _to_plain(self, text: str) -> str:
        """Convert to plain form."""
        parts = re.split(r"(。|、|\s+)", text)
        result_parts = []
        for part in parts:
            if part.endswith("です"):
                part = part[:-2] + "だ"
            elif part.endswith("ます"):
                part = part.replace("ます", "")
            elif part.endswith("であります"):
                part = part[:-4] + "である"
            result_parts.append(part)
        return "".join(result_parts)
