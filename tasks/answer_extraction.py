from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    elif isinstance(value, Mapping):
        for key in [
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
        ]:
            if key in value:
                text = clean_text(value.get(key))
                if text:
                    break
        else:
            text = "\n".join(clean_text(v) for v in value.values())
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        text = "\n".join(part for part in (clean_text(v) for v in value) if part)
    else:
        text = str(value)
    return _strip_wrapper_tokens(" ".join(text.replace("\r", "\n").split())).strip()


def _strip_wrapper_tokens(text: str) -> str:
    patterns = [
        r"<\|begin_of_[^>]+?\|>",
        r"<\|end_of_[^>]+?\|>",
        r"</?think>",
        r"</?analysis>",
        r"</?final>",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def extract_boxed_answer(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return clean_text(boxed[-1])
    return ""


def extract_hash_answer(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    match = re.search(r"####\s*(.+)$", text, flags=re.IGNORECASE)
    return clean_text(match.group(1)) if match else ""


def normalize_math_answer(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = re.sub(r"^\\boxed\{(.+)\}$", r"\1", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    if re.fullmatch(r"[-+]?\d+(?:\.0+)?", text):
        text = re.sub(r"\.0+$", "", text)
    return text


def extract_final_answer(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    patterns = [
        r"Final Answer\s*[:：]\s*(.+)$",
        r"Answer\s*[:：]\s*(.+)$",
        r"Therefore[, ]+the answer is\s+(.+)$",
        r"So[, ]+the answer is\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_math_answer(match.group(1))
    hash_answer = extract_hash_answer(text)
    if hash_answer:
        return normalize_math_answer(hash_answer)
    boxed_answer = extract_boxed_answer(text)
    if boxed_answer:
        return normalize_math_answer(boxed_answer)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return normalize_math_answer(lines[-1] if lines else "")


def split_reasoning_and_final_answer(text: str) -> tuple[str, str]:
    text = clean_text(text)
    if not text:
        return "", ""
    answer = extract_final_answer(text)
    patterns = [
        r"(?is)(.*?)(?:Final Answer|Answer)\s*[:：]\s*(.+)$",
        r"(?is)(.*?)(?:Therefore[, ]+the answer is|So[, ]+the answer is)\s+(.+)$",
        r"(?is)(.*?)####\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            reasoning = clean_text(match.group(1))
            extracted_answer = normalize_math_answer(match.group(2))
            return reasoning or text, extracted_answer or answer
    return text, answer
