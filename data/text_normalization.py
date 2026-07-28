from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


PREFERRED_TEXT_KEYS = (
    "text",
    "content",
    "value",
    "solution",
    "reasoning",
    "response",
    "answer",
    "final_answer",
    "problem",
    "question",
    "prompt",
)

WRAPPER_TOKEN_PATTERNS = (
    r"<\|begin_of_[^>]+?\|>",
    r"<\|end_of_[^>]+?\|>",
    r"</?think>",
    r"</?analysis>",
    r"</?final>",
)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, Mapping):
        for key in PREFERRED_TEXT_KEYS:
            if key in value:
                text = _coerce_text(value.get(key))
                if text:
                    return text
        return "\n".join(part for part in (_coerce_text(item) for item in value.values()) if part)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return "\n".join(part for part in (_coerce_text(item) for item in value) if part)
    return str(value)


def normalize_text(value: Any, *, strip_wrapper_tokens: bool = False) -> str:
    """Normalize transport noise without collapsing semantic whitespace.

    Newline style and trailing horizontal whitespace are normalized. Line
    boundaries, blank lines, and leading indentation are deliberately kept.
    """

    text = _coerce_text(value).replace("\r\n", "\n").replace("\r", "\n")
    if strip_wrapper_tokens:
        for pattern in WRAPPER_TOKEN_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def normalize_inline_text(value: Any, *, strip_wrapper_tokens: bool = False) -> str:
    """Return a single-line representation for fields whose schema requires it."""

    return " ".join(normalize_text(value, strip_wrapper_tokens=strip_wrapper_tokens).split()).strip()
