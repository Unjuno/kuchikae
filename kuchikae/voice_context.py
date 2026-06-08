"""Voice context extraction for Kuchikae v0.1."""

from __future__ import annotations

from kuchikae.types import ProsodyProfile, VoiceContext


class VoiceContextExtractor:
    """Create a VoiceContext from reference audio.

    v0.1 is intentionally model-free: it only marks the context ready when a
    reference audio path is available. Speaker embeddings and real prosody
    extraction are reserved for later backends.
    """

    def extract(self, reference_audio_path: str | None) -> VoiceContext:
        """Return a VoiceContext for the given reference audio path."""
        if not reference_audio_path:
            return VoiceContext(reference_audio_path="", ready=False)

        return VoiceContext(
            reference_audio_path=reference_audio_path,
            ready=True,
            speaker_embedding=None,
            prosody_profile=ProsodyProfile(),
        )
