from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from ..data.text_normalization import normalize_text


class AnswerValidationError(ValueError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)


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
    return normalize_text(text, strip_wrapper_tokens=True)


def _strip_wrapper_tokens(text: str) -> str:
    return normalize_text(text, strip_wrapper_tokens=True)


def extract_boxed_answer(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    answers: list[str] = []
    marker = r"\boxed{"
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            break
        content_start = start + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
            index += 1
        if depth == 0:
            answers.append(text[content_start : index - 1])
            cursor = index
        else:
            cursor = content_start
    return clean_text(answers[-1]) if answers else ""


def extract_hash_answer(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    match = re.search(r"####\s*(.+?)\s*$", text, flags=re.IGNORECASE | re.DOTALL)
    return clean_text(match.group(1)) if match else ""


def normalize_math_answer(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    text = re.sub(r"^(?:the\s+)?answer\s+is\s*[:：]?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    boxed = extract_boxed_answer(text)
    if boxed and re.fullmatch(r"\s*\\boxed\{.*\}\s*", text, flags=re.DOTALL):
        text = boxed
    text = re.sub(r"\s+", " ", text).strip()
    if re.fullmatch(r"[-+]?\d+(?:\.0+)?", text):
        text = re.sub(r"\.0+$", "", text)
    return text


def extract_final_answer(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    hash_answer = extract_hash_answer(text)
    if hash_answer:
        return normalize_math_answer(hash_answer)
    patterns = [
        r"Final\s+Answer\s*[:：]\s*(.+)$",
        r"The\s+answer\s+is\s*[:：]?\s*(.+)$",
        r"Therefore[, ]+the\s+answer\s+is\s*[:：]?\s*(.+)$",
        r"So[, ]+the\s+answer\s+is\s*[:：]?\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return normalize_math_answer(match.group(1))
    boxed_answer = extract_boxed_answer(text)
    if boxed_answer:
        return normalize_math_answer(boxed_answer)
    return ""


def infer_answer_type(answer: str, *, task_type: str = "numeric") -> str:
    answer = normalize_math_answer(answer)
    task_type = str(task_type or "numeric").strip().lower()
    if task_type == "proof":
        return "proof"
    if not answer:
        return "unknown"
    if re.search(r"\\begin\{(?:[pbBvV]?matrix|array)\}", answer):
        return "matrix"
    if (answer.startswith(r"\{") and answer.endswith(r"\}")) or (answer.startswith("{") and answer.endswith("}")):
        return "set"
    if re.fullmatch(r"[\[(]\s*[^,]+\s*,\s*[^,]+\s*[\])]", answer):
        return "interval"
    numeric = r"[-+]?(?:[$£€¥]\s*)?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?"
    if re.fullmatch(numeric, answer):
        return "numeric"
    if re.match(numeric + r"\s+[A-Za-z°][A-Za-z0-9°^/·* ._-]*$", answer):
        return "unit"
    if "\\" in answer or re.search(r"[=+*/^]|[A-Za-z]\s*[-+]\s*\d", answer):
        return "expression"
    if "\n" not in answer and 0 < len(answer.split()) <= 20:
        return "text"
    return "unknown"


def validate_final_answer(
    solution: str,
    final_answer: str,
    *,
    task_type: str = "numeric",
    max_answer_chars: int = 512,
    max_solution_ratio: float = 0.8,
) -> dict[str, str]:
    solution = clean_text(solution)
    answer = normalize_math_answer(final_answer)
    if not answer:
        raise AnswerValidationError("empty_final_answer", "final answer is empty")
    if len(answer) > int(max_answer_chars):
        raise AnswerValidationError("answer_too_long", f"final answer exceeds {max_answer_chars} characters")
    normalized_solution = " ".join(solution.split()).strip()
    normalized_answer = " ".join(answer.split()).strip()
    is_proof = str(task_type or "").strip().lower() == "proof"
    if not is_proof and normalized_solution.casefold() == normalized_answer.casefold():
        raise AnswerValidationError("answer_equals_solution", "non-proof final answer equals the full solution")
    if (
        not is_proof
        and len(normalized_answer) > 64
        and len(normalized_answer) / max(1, len(normalized_solution)) > float(max_solution_ratio)
    ):
        raise AnswerValidationError("answer_solution_ratio_high", "final answer occupies an abnormal fraction of solution")
    answer_type = infer_answer_type(answer, task_type=task_type)
    if answer_type == "unknown":
        raise AnswerValidationError("unknown_answer_type", "final answer type could not be determined")
    return {"final_answer": answer, "answer_type": answer_type}


def split_reasoning_and_final_answer(text: str) -> tuple[str, str]:
    text = clean_text(text)
    if not text:
        return "", ""
    answer = extract_final_answer(text)
    patterns = [
        r"(?is)(.*?)####\s*(.+)$",
        r"(?is)(.*?)Final\s+Answer\s*[:：]\s*(.+)$",
        r"(?is)(.*?)The\s+answer\s+is\s*[:：]?\s*(.+)$",
        r"(?is)(.*?)(?:Therefore[, ]+the\s+answer\s+is|So[, ]+the\s+answer\s+is)\s*[:：]?\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            reasoning = clean_text(match.group(1))
            extracted_answer = normalize_math_answer(match.group(2))
            return reasoning or text, extracted_answer or answer
    boxed_answer = extract_boxed_answer(text)
    if boxed_answer:
        marker = text.rfind(r"\boxed{")
        reasoning = clean_text(text[:marker]) if marker >= 0 else text
        return reasoning or text, normalize_math_answer(boxed_answer)
    return text, answer
