from __future__ import annotations


def test_contract_package_importable() -> None:
    """Verify contract test package imports resolve."""
    from kuchikae.domain import VoiceOutputBackend
    from kuchikae.backends.voice_output import IrodoriTTSVoiceOutputBackend
    assert VoiceOutputBackend is not None
    assert IrodoriTTSVoiceOutputBackend is not None
