from __future__ import annotations


def test_golden_package_importable() -> None:
    """Verify golden test package imports resolve."""
    from kuchikae.domain.text_transform import PromptedRuleTextTransformBackend, RuleTextTransformBackend
    from kuchikae.ui.templates import TEMPLATES
    assert PromptedRuleTextTransformBackend is not None
    assert RuleTextTransformBackend is not None
    assert TEMPLATES is not None
