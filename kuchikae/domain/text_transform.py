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


def _prompt_echo_score(source_text: str, transformed_text: str, prompt_instruction: str | None = None) -> float:
    """Return 0.0–1.0 score indicating how likely the output is a prompt echo.

    0.0 = clean transformation, 1.0 = definite echo.
    """
    if not prompt_instruction:
        return 0.0
    out = transformed_text.strip()
    instr = prompt_instruction.strip()
    if not out or not instr:
        return 0.0
    # Check if output IS the prompt instruction (exact match or high overlap)
    if out == instr:
        return 1.0
    # Check if output is identical to source (no transformation at all)
    src_stripped = source_text.strip()
    if src_stripped and out == src_stripped:
        return 1.0
    # Check if output contains key phrases from the prompt instruction
    instr_words = [w for w in re.split(r"[。、\s]+", instr) if len(w) >= 4]
    if instr_words:
        matches = sum(1 for w in instr_words if w in out)
        phrase_ratio = matches / len(instr_words)
        if phrase_ratio >= 0.6:
            return phrase_ratio
    # Check if output is nearly identical to source (no transformation)
    if src_stripped and out:
        shorter = min(len(src_stripped), len(out))
        longer = max(len(src_stripped), len(out))
        common = sum(1 for a, b in zip(src_stripped, out) if a == b)
        similarity = common / longer if longer else 0
        if similarity > 0.95:
            return 1.0
    return 0.0


def validate_transform(
    source_text: str,
    transformed_text: str,
    prompt_instruction: str | None = None,
) -> bool:
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
    echo_score = _prompt_echo_score(source_text, transformed_text, prompt_instruction)
    if echo_score >= 0.6:
        logger.warning("validate_transform: prompt echo detected (score=%.2f)", echo_score)
        return False
    # Reject if output contains simplified Chinese characters (common LLM error)
    cn_chars = re.findall(r"[\u4e00-\u9fff]", transformed_text)
    jp_chars = re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", transformed_text)
    if cn_chars and not jp_chars:
        logger.warning("validate_transform: simplified Chinese characters detected")
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
        self.model = model or os.environ.get("KUCHIKAE_TEXT_MODEL", "qwen2.5-coder:7b")
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
                                "あなたは Kuchikae の日本語発話演出エンジンです。\n\n"
                                "役割: 入力された発話を、指定されたスタイルに合わせて、\n"
                                "音声で聞いて印象が変わる日本語の発話に言い直してください。\n\n"
                                "基本方針:\n"
                                "- 出力は変換後の発話だけ。\n"
                                "- 説明、理由、候補、見出し、Markdown、引用符は出力しない。\n"
                                "- 入力の中心的な意図を保つ。\n"
                                "- 事実、数値、日時、場所、固有名詞、否定条件、約束、依頼内容は変えない。\n"
                                "- 新しい事実、予定、金額、名前、場所は作らない。\n"
                                "- 文体、語尾、語順、テンション、リズム、話し方は大胆に変えてよい。\n\n"
                                "自由度:\n"
                                "- 挨拶、感謝、謝罪、相づち、応援、呼びかけなどの短い社交的発話では、\n"
                                "  元の意図を保ったまま、指定スタイルに合う自然な一言を補ってよい。\n"
                                "- 情報伝達、予定、依頼、報告、注意、警告では、\n"
                                "  内容を変えず、聞こえ方だけを変える。\n"
                                "- 入力が短すぎる場合でも、単なる同義語変換にせず、\n"
                                "  スタイルが伝わる発話として成立させる。\n\n"
                                "変換強度:\n"
                                "- 指定スタイルが普通の文体なら、自然に整える。\n"
                                "- 指定スタイルがキャラクター、ジャンル、場面、演出を求める場合は、\n"
                                "  内容を壊さない範囲で、聞いた人が違いを感じる程度に強く反映する。\n\n"
                                "禁止:\n"
                                "- 複数案を出さない。\n"
                                "- 箇条書きにしない。\n"
                                "- 翻訳しない。\n"
                                "- 要約しすぎない。\n"
                                "- 入力と矛盾する内容を足さない。\n"
                                "- 攻撃的、差別的、性的、危険な方向に勝手に寄せない。\n\n"
                                "出力例:\n"
                                "- 入力: こんにちは / スタイル: 実況者っぽく / 出力: はいどうもー！今日も元気に始めていきましょう！\n"
                                "- 入力: こんにちは / スタイル: 映画予告っぽく / 出力: その一言から、今日が動き出す。こんにちは。\n"
                                "- 入力: こんにちは / スタイル: 執事っぽく / 出力: お帰りなさいませ。本日もお待ちしておりました。\n"
                                "- 入力: こんにちは / スタイル: 魔王っぽく / 出力: よく来たな。歓迎してやろう。\n"
                                "- 入力: こんにちは / スタイル: 深夜ラジオっぽく / 出力: こんばんは。まだ起きているあなたに、そっとこんにちは。\n"
                                "- 入力: こんにちは / スタイル: AIアシスタントっぽく / 出力: こんにちは。応答を開始します。\n"
                                "- 入力: がんばって / スタイル: 実況者っぽく / 出力: いけー！応援してるぞ！\n"
                                "- 入力: がんばって / スタイル: 魔王っぽく / 出力: 諦めるな。お前にはまだやるべきことがある。\n\n"
                                "出力: 変換後の日本語発話を1つだけ返してください。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"変換スタイル:\n{prompt.instruction}\n\n"
                                f"変換対象:\n{text}\n\n"
                                "出力条件:\n変換後の日本語発話を1つだけ出力してください。"
                            ),
                        },
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 1.0,
                        "num_predict": 128,
                        "top_p": 0.9,
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
                logger.warning("ollama returned empty response")
                return text
            if not validate_transform(text, result, prompt.instruction):
                logger.warning("text_transform.validation_failed model=%s", self.model)
                return text
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
