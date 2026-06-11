"""Text transformation backend interfaces and real implementations."""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from importlib.resources import files
from functools import lru_cache

from kuchikae.domain.types import TextTransformPrompt, TransformState, TransformUpdate

logger = logging.getLogger(__name__)

PROMPT_FILES = {
    "polite": "text_transform_polite.txt",
    "casual": "text_transform_casual.txt",
    "summarize": "text_transform_summarize.txt",
}


class TextTransformBackend(ABC):

    @abstractmethod
    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        raise NotImplementedError


class DummyTextTransformBackend(TextTransformBackend):

    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        instruction = prompt.instruction.strip()
        if not instruction:
            return f"[transformed] {text}"
        return f"[transformed according to prompt] {text}"


class OllamaTextTransformBackend(TextTransformBackend):

    def __init__(self, model: str = "qwen3:8b") -> None:
        self.model = model
        self._base_url = "http://localhost:11434"

    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        import httpx

        logger.info("ollama: model=%s text=%s", self.model, text[:40])
        t0 = time.time()
        try:
            resp = httpx.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "あなたはテキスト変換の専門家です。"
                                "ユーザーから変換ルールと変換対象のテキストが与えられるので、"
                                "ルールに従ってテキストを変換し、**変換結果だけ**を出力してください。\n"
                                "説明や接頭語は一切付けないでください。\n"
                                "元の内容・事実・数値・固有名詞は絶対に変えないでください。\n"
                                "新しい情報を追加しないでください。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"## 変換ルール\n"
                                f"{prompt.instruction}\n\n"
                                f"## 変換対象テキスト\n"
                                f"{text}"
                            ),
                        },
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 256,
                    },
                    "keep_alive": "5m",
                },
                timeout=60,
            )
            resp.raise_for_status()
            body = resp.json()
            msg = body["message"]
            result = (msg.get("content") or msg.get("thinking") or "").strip()
            if not result:
                logger.warning("ollama returned empty response, using original text")
                result = text
            else:
                logger.info("ollama: %.2fs → %s", time.time() - t0, result[:60])
            return result
        except Exception as e:
            logger.warning("ollama failed (%s), falling back to dummy", e)
            return DummyTextTransformBackend().transform(text, prompt)


class GPTTextTransformBackend(TextTransformBackend):

    def __init__(self, model: str = "gpt-oss") -> None:
        self.model = model

    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return DummyTextTransformBackend().transform(text, prompt)

        import httpx

        t0 = time.time()
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
        result = resp.json()["choices"][0]["message"]["content"].strip()
        logger.info("gpt: %.2fs → %s", time.time() - t0, result[:60])
        return result


class RuleTextTransformBackend(TextTransformBackend):

    def __init__(self) -> None:
        self._desu_masu_map = {
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
        if any(kw in instruction for kw in ("丁寧", "です", "ます", "polite")):
            return "desu-masu"
        if any(kw in instruction for kw in (" casual", "カジュアル", "普通形", "plain")):
            return "plain"
        return "desu-masu"

    def _apply(self, text: str, target_form: str) -> str:
        if target_form == "desu-masu":
            return self._to_desu_masu(text)
        return self._to_plain(text)

    def _to_desu_masu(self, text: str) -> str:
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


class PromptedRuleTextTransformBackend(TextTransformBackend):
    """Lightweight backend: rule-based transformation with prompt type detection.
    Prompt templates (with few-shot examples) are loaded for documentation/future LLM use.
    """

    def __init__(self) -> None:
        self._rule_backend = RuleTextTransformBackend()

    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        return self._rule_backend.transform(text, prompt)

    def _detect_prompt_type(self, instruction: str) -> str:
        if any(kw in instruction for kw in ("要約", "まとめ", "summarize")):
            return "summarize"
        if any(kw in instruction for kw in ("カジュアル", "普通形", "タメ口", "casual", "plain")):
            return "casual"
        return "polite"

    @lru_cache(maxsize=4)
    def _load_template(self, prompt_type: str) -> str:
        filename = PROMPT_FILES.get(prompt_type, PROMPT_FILES["polite"])
        try:
            return files("kuchikae.prompts").joinpath(filename).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to load prompt template %s: %s", filename, e)
            return ""


# ---------------------------------------------------------------------------
# Incremental text transform interface
# ---------------------------------------------------------------------------


class IncrementalTextTransformBackend(ABC):
    """Incremental text transform interface.

    Transforms only new (committed) portions of source text while
    preserving previously transformed output.
    """

    @abstractmethod
    def transform_committed(
        self,
        committed_source_text: str,
        previous_state: TransformState,
        prompt: TextTransformPrompt,
    ) -> TransformUpdate:
        ...


class DummyIncrementalTextTransformBackend(IncrementalTextTransformBackend):
    """Dummy incremental transform for testing.

    Applies a simple prefix to each new segment.
    """

    def transform_committed(
        self,
        committed_source_text: str,
        previous_state: TransformState,
        prompt: TextTransformPrompt,
    ) -> TransformUpdate:
        new_text = committed_source_text[previous_state.transformed_up_to:]
        if not new_text:
            return TransformUpdate(
                new_output_segment="",
                updated_state=previous_state,
            )

        instruction = prompt.instruction.strip()
        prefix = "[transformed] " if not instruction else "[transformed according to prompt] "
        new_output_segment = prefix + new_text

        updated_state = TransformState(
            transformed_up_to=len(committed_source_text),
            accumulated_output=previous_state.accumulated_output + new_output_segment,
        )
        return TransformUpdate(
            new_output_segment=new_output_segment,
            updated_state=updated_state,
        )


class TemplateTextTransformBackend(TextTransformBackend):
    """Minimal backend using prompt template substitution only (for testing)."""

    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        instruction = prompt.instruction.strip().lower()
        if any(kw in instruction for kw in ("要約", "まとめ", "summarize")):
            template = self._load_template("summarize")
        elif any(kw in instruction for kw in ("カジュアル", "普通形", "タメ口", "casual", "plain")):
            template = self._load_template("casual")
        else:
            template = self._load_template("polite")

        if "{text}" in template:
            return template.replace("{text}", text)
        return text

    @lru_cache(maxsize=4)
    def _load_template(self, prompt_type: str) -> str:
        filename = PROMPT_FILES.get(prompt_type, PROMPT_FILES["polite"])
        try:
            return files("kuchikae.prompts").joinpath(filename).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to load prompt template %s: %s", filename, e)
            return ""
