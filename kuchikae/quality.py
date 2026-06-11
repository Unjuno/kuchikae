"""Quality heuristic helpers for Kuchikae streaming pipeline.

Checks that transformed text preserves numeric tokens, negation markers,
and other semantically important content from the source.
"""

from __future__ import annotations

import re

# Regex matching common numeric patterns: integers, decimals, percentages,
# date-like numbers (e.g. 2024年, 12月), time (e.g. 14:30), prices (e.g. ¥1200).
_NUMERIC_RE = re.compile(
    r"[+-]?"
    r"(?:\d+(?:[.,]\d+)?|"
    r"[一二三四五六七八九十百千万億兆])"
    r"(?:\s*(?:%|％|年|月|日|時|分|秒|円|ドル|ユーロ|万|億|兆|千|百|十|度|目|人|個))*"
)

# Japanese and English negation markers.
_NEGATION_MARKERS: tuple[str, ...] = (
    "ない", "ません", "ぬ", "ず", "なし", "不要", "禁止", "不可",
    "not", "no ", "n't", "never",
)


def extract_numeric_tokens(text: str) -> list[str]:
    """Return sorted list of unique numeric tokens found in *text*."""
    return sorted(set(_NUMERIC_RE.findall(text)))


def extract_negation_markers(text: str) -> list[str]:
    """Return list of negation markers found in *text* (in order of appearance)."""
    markers: list[str] = []
    lower = text.lower()
    for marker in _NEGATION_MARKERS:
        if marker in lower:
            markers.append(marker)
    return markers


def check_numeric_preservation(
    source_text: str,
    target_text: str,
    *,
    require_exact: bool = False,
) -> bool:
    """Check that numeric tokens in source are preserved in target.

    By default checks that every numeric token in the source
    is present somewhere in the target.  When *require_exact* is
    True, the token sets must be identical.

    Returns True if the check passes.
    """
    source_tokens = extract_numeric_tokens(source_text)
    if not source_tokens:
        return True

    target_tokens = set(extract_numeric_tokens(target_text))

    if require_exact:
        return set(source_tokens) == target_tokens
    return all(token in target_tokens for token in source_tokens)


def check_negation_preservation(source_text: str, target_text: str) -> bool:
    """Check that negation markers in source are preserved in target.

    Returns True if every negation marker found in the source
    is also found in the target.
    """
    source_markers = extract_negation_markers(source_text)
    if not source_markers:
        return True

    lower_target = target_text.lower()
    return all(marker in lower_target for marker in source_markers)
