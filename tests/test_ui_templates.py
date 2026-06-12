from __future__ import annotations

from kuchikae.ui.handlers import TEMPLATES, on_template_change


def test_templates_include_expected_keys() -> None:
    expected = {
        "自然に",
        "魔法使い",
        "勇者への助言",
        "王国の伝令",
        "宇宙船AI",
        "銀河司令官",
        "映画予告",
        "次回予告",
        "ラスボス登場",
        "怪しい占い師",
        "ゲーム実況",
        "通販番組",
        "カスタム",
    }
    assert expected.issubset(TEMPLATES.keys())


def test_non_custom_templates_have_required_phrases() -> None:
    for name, text in TEMPLATES.items():
        if name == "カスタム":
            assert text == ""
            continue
        assert text
        assert "あなたは" in text
        assert "意味を保ったまま" in text
        assert "出力は本文のみ。" in text


def test_on_template_change_custom_returns_empty_update() -> None:
    update = on_template_change("カスタム")
    assert isinstance(update, dict)
    assert update.get("__type__") == "update"


def test_on_template_change_unknown_falls_back_to_natural() -> None:
    update = on_template_change("unknown-template")
    assert isinstance(update, dict)
    assert update.get("value") == TEMPLATES["自然に"]
