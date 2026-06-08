"""VoiceContextExtractor — dummy extractor for v0.1."""

from __future__ import annotations

import uuid

from kuchikae.audio_cache import AudioCache
from kuchikae.types import ProsodyProfile, VoiceContext


class VoiceContextExtractor:
    """Extract a VoiceContext from the reference audio path."""

    def extract(self, cache: AudioCache) -> VoiceContext:
        ref_path = cache.reference_path or ""
        voice_id = str(uuid.uuid4())[:8]
        return VoiceContext(
            voice_id=voice_id,
            reference_audio_path=ref_path,
            ready=True if ref_path else False,
            prosody_profile=ProsodyProfile(),
        )
