"""VoiceContextExtractor — dummy extractor for v0.1."""

from __future__ import annotations

import uuid

from kuchikae.audio_cache import AudioCache
from kuchikae.types import VoiceContext


class VoiceContextExtractor:
    """Extract a VoiceContext from the reference audio path."""

    def extract(self, cache: AudioCache) -> VoiceContext:
        if not cache.reference_path:
            return VoiceContext(voice_id="unknown", ready=False)

        voice_id = str(uuid.uuid4())[:8]
        return VoiceContext(
            voice_id=voice_id,
            reference_path=cache.reference_path,
            ready=True,
        )
