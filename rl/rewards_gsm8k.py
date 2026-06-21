from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Tuple

NUMBER_RE = re.compile(
    r"(?<![\w.])(?:[-+]?\s*\$?\s*|\$?\s*[-+]?\s*)"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?![\w])"
)
HASH_ANSWER_RE = re.compile(r"####\s*(" + NUMBER_RE.pattern + r")")
FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer\s*(?:is|=|:|：)?\s*)(" + NUMBER_RE.pattern + r")",
    flags=re.IGNORECASE,
)
ANSWER_MARKER_RE = re.compile(
    r"(?:^|\n|\s)answer\s*(?:is|=|:|：)\s*(" + NUMBER_RE.pattern + r")",
    flags=re.IGNORECASE,
)


def normalize_number_answer(ans: Any) -> str:
    value = "" if ans is None else str(ans).strip()
    if value.endswith(".") and value.count(".") == 1:
        value = value[:-1]
    value = value.replace("$", "").replace(",", "")
    value = re.sub(r"\s+", "", value)
    return value


def _decimal_or_none(value: Optional[str]) -> Optional[Decimal]:
    if value is None:
        return None
    normalized = normalize_number_answer(value)
    if not normalized:
        return None
    try:
        return Decimal(normalized)
    except (InvalidOperation, ValueError):
        return None


def _numbers_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    pred_dec = _decimal_or_none(pred)
    gold_dec = _decimal_or_none(gold)
    if pred_dec is not None and gold_dec is not None:
        return pred_dec == gold_dec
    return normalize_number_answer(pred) == normalize_number_answer(gold)


def _extract_hash_answer(text: Any) -> Optional[str]:
    matches = HASH_ANSWER_RE.findall(str(text))
    if not matches:
        return None
    return normalize_number_answer(matches[-1])


def _extract_final_answer(text: Any) -> Optional[str]:
    matches = FINAL_ANSWER_RE.findall(str(text))
    if not matches:
        return None
    return normalize_number_answer(matches[-1])


def _extract_answer_marker(text: Any) -> Optional[str]:
    value = str(text)
    matches = list(ANSWER_MARKER_RE.finditer(value))
    if not matches:
        return None
    late_matches = [match for match in matches if match.start() >= max(0, int(len(value) * 0.4))]
    match = (late_matches or matches)[-1]
    return normalize_number_answer(match.group(1))


def _extract_last_number(text: Any) -> Optional[str]:
    matches = NUMBER_RE.findall(str(text))
    if not matches:
        return None
    return normalize_number_answer(matches[-1])


def extract_gsm8k_answer(text: Any) -> Optional[str]:
    hash_answer = _extract_hash_answer(text)
    if hash_answer is not None:
        return hash_answer
    final_answer = _extract_final_answer(text)
    if final_answer is not None:
        return final_answer
    answer_marker = _extract_answer_marker(text)
    if answer_marker is not None:
        return answer_marker
    return _extract_last_number(text)


def _extract_gold_answer(gold_answer: Any) -> Optional[str]:
    hash_answer = _extract_hash_answer(gold_answer)
    if hash_answer is not None:
        return hash_answer
    final_answer = _extract_final_answer(gold_answer)
    if final_answer is not None:
        return final_answer
    answer_marker = _extract_answer_marker(gold_answer)
    if answer_marker is not None:
        return answer_marker
    last_number = _extract_last_number(gold_answer)
    if last_number is not None:
        return last_number
    normalized = normalize_number_answer(gold_answer)
    return normalized or None


def gsm8k_reward(generated_text: str, gold_answer: str) -> Tuple[float, Dict[str, Any]]:
    gold = _extract_gold_answer(gold_answer)
    strict_pred = _extract_hash_answer(generated_text)
    final_answer_pred = _extract_final_answer(generated_text)
    answer_marker_pred = _extract_answer_marker(generated_text)
    fallback_last_number_pred = _extract_last_number(generated_text)
    pred_answer = strict_pred or final_answer_pred or answer_marker_pred or fallback_last_number_pred
    strict_match = _numbers_equal(strict_pred, gold)
    final_answer_match = _numbers_equal(final_answer_pred, gold)
    answer_marker_match = _numbers_equal(answer_marker_pred, gold)
    fallback_match = _numbers_equal(fallback_last_number_pred, gold)
    reward = float(strict_match or final_answer_match or answer_marker_match or fallback_match)
    if strict_match:
        match_type = "strict_hash"
    elif final_answer_match:
        match_type = "final_answer"
    elif answer_marker_match:
        match_type = "answer_marker"
    elif fallback_match:
        match_type = "fallback_last_number"
    else:
        match_type = "none"
    return reward, {
        "pred_answer": pred_answer,
        "gold_answer": gold,
        "strict_answer": strict_pred,
        "final_answer": final_answer_pred,
        "answer_marker_answer": answer_marker_pred,
        "fallback_answer": fallback_last_number_pred,
        "strict_match": bool(strict_match),
        "final_answer_match": bool(final_answer_match),
        "answer_marker_match": bool(answer_marker_match),
        "fallback_match": bool(fallback_match),
        "match_type": match_type,
    }
