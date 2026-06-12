"""Text transformation backend interfaces and real implementations."""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from importlib.resources import files
from functools import lru_cache

from kuchikae.domain.types import TextTransformPrompt, TransformUpdate

logger = logging.getLogger(__name__)

PROMPT_FILES = {
    "polite": "text_transform_polite.txt",
    "casual": "text_transform_casual.txt",
    "summarize": "text_transform_summarize.txt",
}


def strip_cot(text: str) -> str:
    t = text.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    t = re.sub(r"<reasoning>.*?</reasoning>", "", t, flags=re.DOTALL).strip()
    t = re.sub(r"<思考>.*?</思考>", "", t, flags=re.DOTALL).strip()
    return t


def validate_transform(source_text: str, transformed_text: str) -> bool:
    if not transformed_text or not transformed_text.strip():
        return False
    if "<think>" in transformed_text or "</think>" in transformed_text:
        return False
    meta_patterns = (r"^理由[:：]", r"^説明[:：]", r"^候補[:：]", r"^解説[:：]", r"^補足[:：]")
    for pat in meta_patterns:
        if re.match(pat, transformed_text.strip()):
            return False
    max_len = len(source_text) * 3 + 50
    if len(transformed_text) > max_len:
        return False
    src_numbers = set(re.findall(r"\d+", source_text))
    if src_numbers:
        out_numbers = set(re.findall(r"\d+", transformed_text))
        missing = src_numbers - out_numbers
        if missing:
            return False
    return True


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

    def __init__(self, model: str | None = None, strict: bool = False, on_cot_stripped: callable | None = None) -> None:
        self.model = model or os.environ.get("KUCHIKAE_TEXT_MODEL", "qwen2.5:1.5b-instruct")
        self.strict = strict
        self._base_url = os.environ.get("KUCHIKAE_OLLAMA_URL", "http://localhost:11434")
        self._on_cot_stripped = on_cot_stripped

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
                        {"role": "user", "content": f"/no_think\n## 変換ルール\n{prompt.instruction}\n\n## 変換対象テキスト\n{text}"},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 128,
                        "top_p": 0.8,
                    },
                    "keep_alive": "5m",
                },
                timeout=60,
            )
            resp.raise_for_status()
            body = resp.json()
            msg = body["message"]
            result = strip_cot((msg.get("content") or "").strip())
            if "<think>" in (msg.get("content") or "").lower():
                logger.warning("text_transform.cot_stripped model=%s", self.model)
                if self._on_cot_stripped:
                    self._on_cot_stripped(self.model)
            if not result:
                logger.warning("ollama returned empty response, using original text")
                result = text
            if not validate_transform(text, result):
                logger.warning("text_transform.validation_failed model=%s", self.model)
                return text
            else:
                logger.info("ollama: %.2fs → %s", time.time() - t0, result[:60])
            return result
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if self.strict:
                raise RuntimeError(
                    "Ollama text transform is unavailable. "
                    "Start the Ollama server and make sure the requested model is pulled."
                ) from e
            logger.warning("ollama unavailable, falling back to original text: %s", e)
            return text
        except Exception as e:
            if self.strict:
                raise RuntimeError(
                    "Ollama text transform is unavailable. "
                    "Start the Ollama server and make sure the requested model is pulled."
                ) from e
            raise RuntimeError(
                "Ollama text transform failed. Start the Ollama server and make sure the requested model is pulled."
            ) from e


class GPTTextTransformBackend(TextTransformBackend):

    def __init__(self, model: str = "gpt-oss", strict: bool = False) -> None:
        self.model = model
        self.strict = strict

    def transform(self, text: str, prompt: TextTransformPrompt) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            if self.strict:
                raise RuntimeError(
                    "OPENAI_API_KEY is required for GPT text transform."
                )
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
        try:
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"].strip()
            logger.info("gpt: %.2fs → %s", time.time() - t0, result[:60])
            return result
        except Exception as e:
            raise RuntimeError("GPT text transform failed.") from e


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

    Subclasses maintain internal per-session state.
    """

    @abstractmethod
    def transform_committed(
        self,
        committed_source_text: str,
        prompt: TextTransformPrompt,
        session_id: str = "",
    ) -> TransformUpdate:
        ...


class DummyIncrementalTextTransformBackend(IncrementalTextTransformBackend):
    """Dummy incremental transform for testing.

    Applies a simple prefix to each new segment.
    Per-session state is tracked internally.
    """

    def __init__(self) -> None:
        # per-session state: last transformed length so far
        self._transformed_up_to: dict[str, int] = {}
        self._accumulated: dict[str, str] = {}

    def transform_committed(
        self,
        committed_source_text: str,
        prompt: TextTransformPrompt,
        session_id: str = "",
    ) -> TransformUpdate:
        prev = self._transformed_up_to.get(session_id, 0)
        accumulated = self._accumulated.get(session_id, "")
        new_text = committed_source_text[prev:]
        if not new_text:
            return TransformUpdate(
                session_id=session_id,
                source_committed_text=committed_source_text,
                transformed_committed_text=accumulated,
                newly_transformed_text="",
                is_final=False,
            )

        instruction = prompt.instruction.strip()
        prefix = "[transformed] " if not instruction else "[transformed according to prompt] "
        newly_transformed = prefix + new_text
        accumulated += newly_transformed
        self._transformed_up_to[session_id] = len(committed_source_text)
        self._accumulated[session_id] = accumulated

        return TransformUpdate(
            session_id=session_id,
            source_committed_text=committed_source_text,
            transformed_committed_text=accumulated,
            newly_transformed_text=newly_transformed,
            is_final=False,
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
