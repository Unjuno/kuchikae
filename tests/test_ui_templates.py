from __future__ import annotations

from kuchikae.ui.handlers import TEMPLATES, on_template_change


def test_templates_include_expected_keys() -> None:
    expected = {
        "自然に",
        "丁寧に",
        "親しみやすく",
        "短く",
        "力強く",
        "落ち着いて",
        "実況者っぽく",
        "映画予告っぽく",
        "AIアシスタントっぽく",
        "執事っぽく",
        "魔王っぽく",
        "深夜ラジオっぽく",
        "カスタム",
    }
    assert expected.issubset(TEMPLATES.keys())


def test_non_custom_templates_have_required_phrases() -> None:
    for name, text in TEMPLATES.items():
        if name == "カスタム":
            assert text == ""
            continue
        assert text
        assert len(text) > 10


def test_performance_templates_exist() -> None:
    assert "実況者っぽく" in TEMPLATES
    assert "映画予告っぽく" in TEMPLATES
    assert "AIアシスタントっぽく" in TEMPLATES
    assert "執事っぽく" in TEMPLATES
    assert "魔王っぽく" in TEMPLATES
    assert "深夜ラジオっぽく" in TEMPLATES


def test_on_template_change_custom_returns_empty_update() -> None:
    update = on_template_change("カスタム")
    assert isinstance(update, dict)
    assert update.get("__type__") == "update"


def test_on_template_change_unknown_falls_back_to_natural() -> None:
    update = on_template_change("unknown-template")
    assert isinstance(update, dict)
    assert update.get("value") == TEMPLATES["自然に"]
