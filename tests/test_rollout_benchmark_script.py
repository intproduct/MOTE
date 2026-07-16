from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_rollout_backends.py"
SPEC = importlib.util.spec_from_file_location("benchmark_rollout_backends", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_response_length_stops_at_eos_and_padding():
    assert MODULE._response_length([10, 11, 2, 2], eos_token_id=2, pad_token_id=2) == 3
    assert MODULE._response_length([10, 11, 0, 0], eos_token_id=2, pad_token_id=0) == 2


def test_metric_summary_uses_medians():
    result = MODULE._metric_summary([2.0, 1.0, 4.0], [20, 20, 20], row_count=10)
    assert result["median_generate_seconds"] == 2.0
    assert result["median_prompts_per_second"] == 5.0
    assert result["median_generated_tokens_per_second"] == 10.0


def test_build_comparison_reports_vllm_speedup():
    result = MODULE.build_comparison(
        {
            "hf": {
                "median_prompts_per_second": 5.0,
                "median_generated_tokens_per_second": 100.0,
                "load_seconds": 2.0,
            },
            "vllm": {
                "median_prompts_per_second": 15.0,
                "median_generated_tokens_per_second": 250.0,
                "load_seconds": 5.0,
            },
        }
    )
    assert result["vllm_vs_hf_prompts_per_second_speedup"] == 3.0
    assert result["vllm_vs_hf_generated_tokens_per_second_speedup"] == 2.5
    assert result["vllm_minus_hf_load_seconds"] == 3.0


def test_validate_config_rejects_missing_sections():
    with pytest.raises(ValueError, match="missing config section"):
        MODULE._validate_config({"model": {"path": "/tmp/model"}})
