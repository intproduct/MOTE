from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


NUMBER_RE = re.compile(
    r"(?<![\w.])(?:[-+]?\s*\$?\s*|\$?\s*[-+]?\s*)"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?![\w])"
)
HASH_ANSWER_RE = re.compile(r"####\s*(" + NUMBER_RE.pattern + r")")


def normalize_gsm8k_number(x: str) -> str:
    value = str(x).strip()
    value = value.rstrip()
    if value.endswith(".") and value.count(".") == 1:
        value = value[:-1]
    value = value.replace("$", "").replace(",", "")
    value = re.sub(r"\s+", "", value)
    return value


def _decimal_or_none(value: str) -> Optional[Decimal]:
    normalized = normalize_gsm8k_number(value)
    if not normalized:
        return None
    try:
        return Decimal(normalized)
    except (InvalidOperation, ValueError):
        return None


def numbers_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    pred_dec = _decimal_or_none(pred)
    gold_dec = _decimal_or_none(gold)
    if pred_dec is not None and gold_dec is not None:
        return pred_dec == gold_dec
    return normalize_gsm8k_number(pred) == normalize_gsm8k_number(gold)


def extract_hash_answer(text: str) -> Optional[str]:
    matches = HASH_ANSWER_RE.findall(str(text))
    if not matches:
        return None
    return normalize_gsm8k_number(matches[-1])


def extract_last_number(text: str) -> Optional[str]:
    matches = NUMBER_RE.findall(str(text))
    if not matches:
        return None
    return normalize_gsm8k_number(matches[-1])


def extract_gold_answer(answer: str) -> str:
    hash_answer = extract_hash_answer(answer)
    if hash_answer is not None:
        return hash_answer
    last_number = extract_last_number(answer)
    if last_number is not None:
        return last_number
    return normalize_gsm8k_number(answer)


def _first_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _first_text(item)
            if text is not None:
                return text
    return None


def extract_lm_eval_continuation(sample: Dict[str, Any]) -> Optional[str]:
    """Return only the model continuation from lm_eval sample logs."""
    return _first_text(sample.get("resps"))


def _sample_doc(sample: Dict[str, Any]) -> Dict[str, Any]:
    doc = sample.get("doc")
    return doc if isinstance(doc, dict) else {}


def _sample_question(sample: Dict[str, Any]) -> Optional[str]:
    doc = _sample_doc(sample)
    for key in ("question", "problem", "input", "query"):
        value = doc.get(key)
        if isinstance(value, str):
            return value
    return None


def _sample_gold_raw(sample: Dict[str, Any]) -> Optional[str]:
    doc = _sample_doc(sample)
    for key in ("answer", "target", "gold", "gold_answer"):
        value = doc.get(key)
        if isinstance(value, str):
            return value
    target = sample.get("target")
    if isinstance(target, str):
        return target
    return None


def _first_filtered_response(sample: Dict[str, Any]) -> Optional[str]:
    return _first_text(sample.get("filtered_resps"))


def _filter_name(sample: Dict[str, Any]) -> str:
    return str(sample.get("filter") or "")


def build_gsm8k_metric_samples(task_name: str, lm_eval_samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_doc: Dict[Any, Dict[str, Any]] = {}
    for idx, sample in enumerate(lm_eval_samples):
        doc_id = sample.get("doc_id", idx)
        record = by_doc.setdefault(
            doc_id,
            {
                "task": task_name,
                "sample_id": len(by_doc),
                "doc_id": doc_id,
                "question": _sample_question(sample),
                "gold_answer_raw": _sample_gold_raw(sample),
                "raw_output": extract_lm_eval_continuation(sample),
                "lm_eval_strict_prediction": None,
                "lm_eval_flexible_prediction": None,
            },
        )
        if record.get("question") is None:
            record["question"] = _sample_question(sample)
        if record.get("gold_answer_raw") is None:
            record["gold_answer_raw"] = _sample_gold_raw(sample)
        if record.get("raw_output") is None:
            record["raw_output"] = extract_lm_eval_continuation(sample)

        filter_name = _filter_name(sample)
        filtered = _first_filtered_response(sample)
        if filter_name == "strict-match":
            record["lm_eval_strict_prediction"] = filtered
        elif filter_name == "flexible-extract":
            record["lm_eval_flexible_prediction"] = filtered

    return list(by_doc.values())


def _rate(count: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return float(count) / float(total)


def compute_gsm8k_fallback_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    computable = 0
    strict_correct_count = 0
    fallback_correct_count = 0
    last_number_correct_count = 0
    has_hash_count = 0
    correct_but_no_hash_count = 0
    strict_correct_fallback_wrong_count = 0
    fallback_correct_strict_wrong_count = 0

    for idx, sample in enumerate(samples):
        raw_output = sample.get("raw_output")
        gold_raw = sample.get("gold_answer_raw")
        debug_flags: List[str] = []
        if not isinstance(raw_output, str):
            debug_flags.append("missing_continuation_from_resps")
        if not isinstance(gold_raw, str):
            debug_flags.append("missing_gold_answer")

        gold_answer = extract_gold_answer(gold_raw) if isinstance(gold_raw, str) else None
        hash_answer = extract_hash_answer(raw_output) if isinstance(raw_output, str) else None
        last_number = extract_last_number(raw_output) if isinstance(raw_output, str) else None
        has_hash_answer = hash_answer is not None
        fallback_answer = hash_answer if hash_answer is not None else last_number

        strict_hash_correct = numbers_equal(hash_answer, gold_answer)
        last_number_correct = numbers_equal(last_number, gold_answer)
        fallback_correct = numbers_equal(fallback_answer, gold_answer)
        correct_but_no_hash = bool((not has_hash_answer) and last_number_correct)
        strict_correct_fallback_wrong = bool(strict_hash_correct and not fallback_correct)
        fallback_correct_strict_wrong = bool(fallback_correct and not strict_hash_correct)

        if strict_correct_fallback_wrong:
            debug_flags.append("strict_correct_fallback_wrong")
        if fallback_correct_strict_wrong:
            debug_flags.append("fallback_correct_strict_wrong")

        is_computable = isinstance(raw_output, str) and isinstance(gold_raw, str)
        if is_computable:
            computable += 1
            strict_correct_count += int(strict_hash_correct)
            fallback_correct_count += int(fallback_correct)
            last_number_correct_count += int(last_number_correct)
            has_hash_count += int(has_hash_answer)
            correct_but_no_hash_count += int(correct_but_no_hash)
            strict_correct_fallback_wrong_count += int(strict_correct_fallback_wrong)
            fallback_correct_strict_wrong_count += int(fallback_correct_strict_wrong)

        records.append(
            {
                "sample_id": sample.get("sample_id", idx),
                "doc_id": sample.get("doc_id"),
                "task": sample.get("task"),
                "question": sample.get("question"),
                "gold_answer_raw": gold_raw,
                "gold_answer": gold_answer,
                "raw_output": raw_output,
                "lm_eval_strict_prediction": sample.get("lm_eval_strict_prediction"),
                "lm_eval_flexible_prediction": sample.get("lm_eval_flexible_prediction"),
                "hash_answer": hash_answer,
                "last_number": last_number,
                "fallback_answer": fallback_answer,
                "has_hash_answer": has_hash_answer,
                "hash_answer_correct": strict_hash_correct,
                "strict_correct": strict_hash_correct,
                "strict_hash_correct": strict_hash_correct,
                "last_number_correct": last_number_correct,
                "fallback_correct": fallback_correct,
                "correct_but_no_hash": correct_but_no_hash,
                "strict_correct_fallback_wrong": strict_correct_fallback_wrong,
                "fallback_correct_strict_wrong": fallback_correct_strict_wrong,
                "debug_flags": debug_flags,
            }
        )

    strict_hash_acc = _rate(strict_correct_count, computable)
    fallback_last_number_acc = _rate(fallback_correct_count, computable)
    fallback_minus_strict = (
        None
        if strict_hash_acc is None or fallback_last_number_acc is None
        else fallback_last_number_acc - strict_hash_acc
    )

    if fallback_minus_strict is not None and fallback_minus_strict < 0:
        warnings.append(
            "GSM8K fallback_last_number_acc is lower than strict_hash_acc; this violates the custom parser invariant."
        )
    if strict_correct_fallback_wrong_count:
        warnings.append(
            "GSM8K strict_correct_fallback_wrong_count is non-zero; inspect samples with debug_flags."
        )
    missing_continuations = sum(1 for record in records if "missing_continuation_from_resps" in record["debug_flags"])
    if missing_continuations:
        warnings.append(
            f"GSM8K custom metrics skipped {missing_continuations} sample(s) without a continuation in lm_eval resps."
        )

    summary = {
        "strict_hash_acc": strict_hash_acc,
        "fallback_last_number_acc": fallback_last_number_acc,
        "last_number_acc": _rate(last_number_correct_count, computable),
        "has_hash_answer_rate": _rate(has_hash_count, computable),
        "correct_but_no_hash_rate": _rate(correct_but_no_hash_count, computable),
        "fallback_minus_strict": fallback_minus_strict,
        "strict_correct_fallback_wrong_count": strict_correct_fallback_wrong_count,
        "fallback_correct_strict_wrong_count": fallback_correct_strict_wrong_count,
        "num_samples": computable,
        "num_total_samples": len(records),
        "warnings": warnings,
    }
    return {"summary": summary, "records": records, "warnings": warnings}


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
