"""Deterministic, language-preserving evaluation text normalization."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize for scoring without translating or standardizing lexical forms.

    The pipeline applies NFC, Unicode-aware lowercasing, replaces every Unicode
    punctuation character with a space, and collapses whitespace. Replacing
    punctuation (instead of deleting it) avoids accidentally joining words.
    """
    normalized = unicodedata.normalize("NFC", text).lower()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return _WHITESPACE.sub(" ", without_punctuation).strip()
