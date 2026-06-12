from __future__ import annotations

from kuchikae.ui.handlers import TEMPLATES, on_template_change


def test_templates_include_expected_keys() -> None:
    expected = {
        "自然に",
        "丁寧に",
        "簡潔に",
        "親しみやすく",
        "かっこよく",
        "ユーモラスに",
        "力強く",
        "落ち着いて",
        "カスタム",
    }
    assert expected.issubset(TEMPLATES.keys())


def test_non_custom_templates_have_required_phrases() -> None:
    for name, text in TEMPLATES.items():
        if name == "カスタム":
            assert text == ""
            continue
        assert text
        assert "意味を保ったまま" in text
        assert "出力は変換後の文章のみ。" in text


def test_on_template_change_custom_returns_empty_update() -> None:
    update = on_template_change("カスタム")
    assert isinstance(update, dict)
    assert update.get("__type__") == "update"


def test_on_template_change_unknown_falls_back_to_natural() -> None:
    update = on_template_change("unknown-template")
    assert isinstance(update, dict)
    assert update.get("value") == TEMPLATES["自然に"]
