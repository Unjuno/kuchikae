"""Audio key for performance caching."""

from __future__ import annotations

import os

from kuchikae.types import AudioCacheKey


class AudioKey:
    """Audio key for performance caching.

    first implementation may use path + size + mtime
    do not hash large files unless necessary
    provide tests
    """

    def __init__(self, path: str, size: int, mtime: float) -> None:
        self.path = path
        self.size = size
        self.mtime = mtime

    @classmethod
    def from_file(cls, path: str) -> "AudioKey":
        """Create AudioKey from file."""
        st = os.stat(path)
        return cls(path=os.path.abspath(path), size=st.st_size, mtime=st.st_mtime)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AudioKey):
            return False
        return (
            self.path == other.path
            and self.size == other.size
            and self.mtime == other.mtime
        )

    def __hash__(self) -> int:
        return hash((self.path, self.size, self.mtime))

    def __repr__(self) -> str:
        return f"AudioKey(path={self.path!r}, size={self.size}, mtime={self.mtime})"


class AudioKeyFromCacheKey(AudioKey):
    """Adapter to convert AudioCacheKey to AudioKey."""

    def __init__(self, cache_key: AudioCacheKey) -> None:
        super().__init__(cache_key.path, cache_key.size, cache_key.mtime)


class AudioKeyFromPath:
    """Audio key from path only (less precise)."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AudioKeyFromPath):
            return False
        return self.path == other.path

    def __hash__(self) -> int:
        return hash(self.path)

    def __repr__(self) -> str:
        return f"AudioKeyFromPath(path={self.path!r})"
