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
        "先生っぽく",
        "友達っぽく",
        "ニュースキャスターっぽく",
        "セールスっぽく",
        "詩的に",
        "実験: 関西弁",
        "実験: ギャルっぽく",
        "実験: 赤ちゃんっぽく",
        "実験: 武士っぽく",
        "実験: 毒舌",
        "実験: 皮肉っぽく",
        "実験: 外国人っぽく",
        "実験: 特定キャラっぽく",
        "実験強: 関西弁",
        "実験強: ギャルっぽく",
        "実験強: 赤ちゃんっぽく",
        "実験強: 武士っぽく",
        "実験強: 毒舌",
        "実験強: 皮肉っぽく",
        "実験強: 外国人っぽく",
        "実験強: 特定キャラっぽく",
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
    update, warning = on_template_change("カスタム")
    assert isinstance(update, dict)
    assert update.get("__type__") == "update"
    assert warning == ""


def test_on_template_change_unknown_falls_back_to_natural() -> None:
    update, warning = on_template_change("unknown-template")
    assert isinstance(update, dict)
    assert update.get("value") == TEMPLATES["自然に"]
    assert warning == ""


def test_official_candidate_templates_exist() -> None:
    official_candidates = [
        "先生っぽく",
        "友達っぽく",
        "ニュースキャスターっぽく",
        "セールスっぽく",
        "詩的に",
    ]
    for template in official_candidates:
        assert template in TEMPLATES, f"Official candidate template '{template}' not found"


def test_experimental_candidate_templates_exist() -> None:
    experimental_candidates = [
        "実験: 関西弁",
        "実験: ギャルっぽく",
        "実験: 赤ちゃんっぽく",
        "実験: 武士っぽく",
        "実験: 毒舌",
        "実験: 皮肉っぽく",
        "実験: 外国人っぽく",
        "実験: 特定キャラっぽく",
    ]
    for template in experimental_candidates:
        assert template in TEMPLATES, f"Experimental candidate template '{template}' not found"


def test_strong_experimental_candidate_templates_exist() -> None:
    strong_experimental_candidates = [
        "実験強: 関西弁",
        "実験強: ギャルっぽく",
        "実験強: 赤ちゃんっぽく",
        "実験強: 武士っぽく",
        "実験強: 毒舌",
        "実験強: 皮肉っぽく",
        "実験強: 外国人っぽく",
        "実験強: 特定キャラっぽく",
    ]
    for template in strong_experimental_candidates:
        assert template in TEMPLATES, f"Strong experimental candidate template '{template}' not found"


def test_experimental_templates_have_style_template_markers() -> None:
    for name in TEMPLATES:
        if name.startswith("実験: ") or name.startswith("実験強: "):
            assert "[STYLE_TEMPLATE: " in TEMPLATES[name], f"Template '{name}' missing [STYLE_TEMPLATE: ...] marker"
