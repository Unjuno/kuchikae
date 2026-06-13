from __future__ import annotations

from unittest.mock import patch

from kuchikae.domain.text_transform import OllamaTextTransformBackend, strip_cot, validate_transform
from kuchikae.domain.types import TextTransformPrompt


def test_strip_cot_removes_tags() -> None:
    assert strip_cot("a <think>b</think> c") == "a  c".strip()


def test_validate_transform_rejects_meta_and_numeric_loss() -> None:
    assert not validate_transform("今日は3件", "")
    assert not validate_transform("今日は3件", "理由: 今日は3件です")
    assert not validate_transform("今日は3件", "今日は")


def test_validate_transform_allows_short_kanji_only_japanese() -> None:
    assert validate_transform("ありがとう", "大丈夫")
    assert validate_transform("ありがとう", "了解")
    assert validate_transform("ありがとう", "承知")
    assert validate_transform("ありがとう", "感謝")
    assert validate_transform("ありがとう", "無理")
    assert validate_transform("ありがとう", "確認")
    assert validate_transform("ありがとう", "失礼")
    assert validate_transform("ありがとう", "了解です")
    assert validate_transform("ありがとう", "承知しました")


def test_validate_transform_rejects_cjk_only_long_output() -> None:
    assert not validate_transform("こんにちは", "你好，今天也请多关照")
    assert not validate_transform("ありがとう", "感谢您的帮助和支持非常感谢")


def test_validate_transform_allows_cjk_with_kana() -> None:
    assert validate_transform("こんにちは", "こんにちは、今日はいい天気ですね")


def test_ollama_transform_empty_result_returns_empty(monkeypatch) -> None:
    backend = OllamaTextTransformBackend(model="dummy", strict=False)

    class FakeResp:
        def raise_for_status(self):
            return None
        def json(self):
            return {"message": {"content": ""}}

    with patch("httpx.post", return_value=FakeResp()):
        out = backend.transform("hello", TextTransformPrompt(instruction="polite"))
    assert out == ""


def test_ollama_transform_validation_failed_returns_result(monkeypatch) -> None:
    backend = OllamaTextTransformBackend(model="dummy", strict=False)

    class FakeResp:
        def raise_for_status(self):
            return None
        def json(self):
            return {"message": {"content": "理由: テスト"}}

    with patch("httpx.post", return_value=FakeResp()):
        out = backend.transform("hello", TextTransformPrompt(instruction="polite"))
    assert out == "理由: テスト"


def test_ollama_transform_uses_content_not_thinking(monkeypatch) -> None:
    backend = OllamaTextTransformBackend(model="dummy", strict=False)

    class FakeResp:
        def raise_for_status(self):
            return None
        def json(self):
            return {"message": {"content": "ok", "thinking": "bad"}}

    with patch("httpx.post", return_value=FakeResp()):
        out = backend.transform("hello", TextTransformPrompt(instruction="polite"))
    assert out == "ok"


def test_extract_template_name_with_experimental_marker():
    from kuchikae.domain.text_transform import _extract_template_name
    assert _extract_template_name("[STYLE_TEMPLATE: 実験: 関西弁]") == "実験: 関西弁"
    assert _extract_template_name("[STYLE_TEMPLATE: 実験: 毒舌]") == "実験: 毒舌"
    assert _extract_template_name("[STYLE_TEMPLATE: 実験強: 関西弁]") == "実験強: 関西弁"
    assert _extract_template_name("[STYLE_TEMPLATE: 実験強: 毒舌]") == "実験強: 毒舌"


def test_extract_template_name_without_marker():
    from kuchikae.domain.text_transform import _extract_template_name
    assert _extract_template_name("普通のテンプレート") is None
    assert _extract_template_name("") is None
    assert _extract_template_name(None) is None


def test_build_few_shot_block_returns_content_for_known_template():
    from kuchikae.domain.text_transform import _build_few_shot_block
    block = _build_few_shot_block("実験強: 関西弁")
    assert block
    assert "関西弁" in block
    assert "入力:" in block
    assert "出力:" in block


def test_build_few_shot_block_returns_empty_for_none():
    from kuchikae.domain.text_transform import _build_few_shot_block
    assert _build_few_shot_block(None) == ""


def test_build_few_shot_block_returns_empty_for_unknown():
    from kuchikae.domain.text_transform import _build_few_shot_block
    assert _build_few_shot_block("存在しないテンプレート") == ""


def test_experimental_dokuji_few_shot_no_aggressive_commands():
    from kuchikae.domain.text_transform import STYLE_FEW_SHOTS
    examples = STYLE_FEW_SHOTS["実験: 毒舌"]
    for inp, out in examples:
        if "資料を送ってください" in inp:
            assert "送れ" not in out, f"実験: 毒舌 should not have aggressive command: {out}"
            assert "遅れるな" not in out, f"実験: 毒舌 should not have aggressive command: {out}"
        if "田中さんに確認してください" in inp:
            assert "しろ" not in out, f"実験: 毒舌 should not have aggressive command: {out}"
            assert "俺には聞くな" not in out, f"実験: 毒舌 should not have aggressive command: {out}"


def test_experimental_hinniku_few_shot_no_overly_strong():
    from kuchikae.domain.text_transform import STYLE_FEW_SHOTS
    examples = STYLE_FEW_SHOTS["実験: 皮肉っぽく"]
    for inp, out in examples:
        if "資料を送ってください" in inp:
            assert "期待はしていません" not in out, f"実験: 皮肉っぽく should not be overly strong: {out}"
        if "田中さんに確認してください" in inp:
            assert "確率は低い" not in out, f"実験: 皮肉っぽく should not be overly strong: {out}"


def test_strong_experimental_dokuji_few_shot_has_aggressive():
    from kuchikae.domain.text_transform import STYLE_FEW_SHOTS
    examples = STYLE_FEW_SHOTS["実験強: 毒舌"]
    found = False
    for inp, out in examples:
        if "資料を送ってください" in inp:
            found = True
            assert "送れ" in out or "遅れるな" in out, f"実験強: 毒舌 should have aggressive examples: {out}"
    assert found, "実験強: 毒舌 should have a 資料送付 example"

