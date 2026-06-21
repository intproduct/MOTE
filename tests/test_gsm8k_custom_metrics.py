from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg
if "MOTE.eval" not in sys.modules:
    eval_pkg = types.ModuleType("MOTE.eval")
    eval_pkg.__path__ = [str(ROOT / "eval")]
    sys.modules["MOTE.eval"] = eval_pkg

from MOTE.eval import gsm8k_custom_metrics as gsm


def test_extract_hash_answer():
    assert gsm.extract_hash_answer("#### 42") == "42"
    assert gsm.extract_hash_answer("####42") == "42"
    assert gsm.extract_hash_answer("#### -1,234.50") == "-1234.50"
    assert gsm.extract_hash_answer("first #### 1 then final #### 2") == "2"


def test_extract_last_number():
    assert gsm.extract_last_number("The answer is 12. Then final is 42.") == "42"
    assert gsm.extract_last_number("Final answer: $1,234") == "1234"
    assert gsm.extract_last_number("No number here") is None


def test_fallback_prefers_hash():
    result = gsm.compute_gsm8k_fallback_metrics(
        [
            {
                "sample_id": 0,
                "task": "gsm8k",
                "gold_answer_raw": "#### 41",
                "raw_output": "The formatted answer is #### 41. A trailing debug number is 42.",
            }
        ]
    )
    record = result["records"][0]
    assert record["hash_answer"] == "41"
    assert record["last_number"] == "42"
    assert record["fallback_answer"] == "41"
    assert record["strict_hash_correct"] is True
    assert record["fallback_correct"] is True


def test_fallback_uses_last_number_when_no_hash():
    result = gsm.compute_gsm8k_fallback_metrics(
        [
            {
                "sample_id": 0,
                "task": "gsm8k",
                "gold_answer_raw": "#### 42",
                "raw_output": "No hash marker here. The final answer is 42.",
            }
        ]
    )
    record = result["records"][0]
    assert record["strict_hash_correct"] is False
    assert record["fallback_correct"] is True
    assert record["correct_but_no_hash"] is True
    assert result["summary"]["fallback_last_number_acc"] == 1.0
    assert result["summary"]["strict_hash_acc"] == 0.0


def test_fallback_is_not_lm_eval_flexible():
    # This collaborator-style fallback is intentionally not lm_eval's flexible-extract.
    sample = {
        "doc_id": 7,
        "filter": "flexible-extract",
        "doc": {"question": "q", "answer": "#### 42"},
        "resps": [["We tried 10. Final answer: 42."]],
        "filtered_resps": ["lm_eval_flexible_value"],
    }
    metric_samples = gsm.build_gsm8k_metric_samples("gsm8k", [sample])
    result = gsm.compute_gsm8k_fallback_metrics(metric_samples)
    assert result["records"][0]["lm_eval_flexible_prediction"] == "lm_eval_flexible_value"
    assert result["records"][0]["fallback_answer"] == "42"


def test_number_normalization():
    assert gsm.numbers_equal("42", "42.0")
    assert gsm.numbers_equal("1,000", "1000")
    assert gsm.numbers_equal("$5", "5")


def test_invariant_warning_when_strict_correct_fallback_wrong_count_nonzero():
    with patch.object(gsm, "numbers_equal", side_effect=[True, False, False]):
        result = gsm.compute_gsm8k_fallback_metrics(
            [
                {
                    "sample_id": 0,
                    "task": "gsm8k",
                    "gold_answer_raw": "#### 42",
                    "raw_output": "#### 42",
                }
            ]
        )
    assert result["summary"]["strict_correct_fallback_wrong_count"] == 1
    assert result["summary"]["fallback_last_number_acc"] < result["summary"]["strict_hash_acc"]
    assert result["warnings"]
    assert "strict_correct_fallback_wrong" in result["records"][0]["debug_flags"]


def test_missing_continuation_from_resps_is_not_prompt_fallback():
    metric_samples = gsm.build_gsm8k_metric_samples(
        "gsm8k",
        [
            {
                "doc_id": 0,
                "filter": "strict-match",
                "doc": {"question": "Prompt has number 123", "answer": "#### 42"},
                "arguments": ["few-shot prompt with #### 42"],
                "filtered_resps": ["42"],
            }
        ],
    )
    result = gsm.compute_gsm8k_fallback_metrics(metric_samples)
    assert result["records"][0]["raw_output"] is None
    assert "missing_continuation_from_resps" in result["records"][0]["debug_flags"]
    assert result["summary"]["num_samples"] == 0
