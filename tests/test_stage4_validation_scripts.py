from __future__ import annotations

import importlib.util
from pathlib import Path
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_stage4_rl_run", ROOT / "scripts" / "validate_stage4_rl_run.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CUDA_SPEC = importlib.util.spec_from_file_location("validate_stage4_cuda", ROOT / "scripts" / "validate_stage4_cuda.py")
CUDA_MODULE = importlib.util.module_from_spec(CUDA_SPEC)
assert CUDA_SPEC.loader is not None
CUDA_SPEC.loader.exec_module(CUDA_MODULE)


def test_cuda_validator_effective_prompt_ids_strip_padding():
    encoded = {
        "input_ids": torch.tensor([[0, 0, 4, 5], [6, 7, 8, 9]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
    }
    assert CUDA_MODULE._effective_prompt_ids(encoded) == [[4, 5], [6, 7, 8, 9]]


def test_stage4_rl_run_validator_accepts_fresh_finite_updates():
    records = [
        {"kind": "run_start"},
        {"kind": "train", "update_step": 1, "did_update": True, "loss": 1.0, "policy_loss": 0.5, "policy_lag_updates": 0, "vllm_policy_version": 1},
        {"kind": "train", "update_step": 2, "did_update": True, "loss": 0.8, "policy_loss": 0.4, "policy_lag_updates": 0, "vllm_policy_version": 2},
    ]
    result = MODULE.validate_records(records, min_updates=2, allow_fallback=False)
    assert result["ok"] is True


def test_stage4_rl_run_validator_rejects_lag_and_fallback():
    records = [
        {"kind": "run_start"},
        {"kind": "train", "update_step": 1, "did_update": True, "loss": 1.0, "policy_loss": 0.5, "policy_lag_updates": 1, "vllm_policy_version": 0, "vllm_fallback_used": True},
    ]
    result = MODULE.validate_records(records, min_updates=1, allow_fallback=False)
    assert result["ok"] is False
    assert any("policy_lag" in error for error in result["errors"])
    assert any("fallback" in error for error in result["errors"])
