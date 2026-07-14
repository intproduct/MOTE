from __future__ import annotations

import importlib.util
import json
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
SYNC_SPEC = importlib.util.spec_from_file_location(
    "validate_stage4_sync_artifacts", ROOT / "scripts" / "validate_stage4_sync_artifacts.py"
)
SYNC_MODULE = importlib.util.module_from_spec(SYNC_SPEC)
assert SYNC_SPEC.loader is not None
SYNC_SPEC.loader.exec_module(SYNC_MODULE)


def test_cuda_validator_effective_prompt_ids_strip_padding():
    encoded = {
        "input_ids": torch.tensor([[0, 0, 4, 5], [6, 7, 8, 9]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
    }
    assert CUDA_MODULE._effective_prompt_ids(encoded) == [[4, 5], [6, 7, 8, 9]]


def test_stage4_rl_run_validator_accepts_fresh_finite_updates():
    records = [
        {"kind": "run_start", "vllm_policy_fingerprint": "a"},
        {"kind": "train", "update_step": 1, "did_update": True, "loss": 1.0, "policy_loss": 0.5, "policy_lag_updates": 0, "vllm_policy_version": 0, "vllm_post_update_policy_version": 1, "vllm_policy_fingerprint": "a", "vllm_post_update_policy_fingerprint": "b", "rollout_request_id": "u0-r1", "rollout_prompt_fingerprint": "p1", "rollout_sampling_fingerprint": "s", "vllm_engine_policy_verified": True, "vllm_export_transaction_committed": True},
        {"kind": "train", "update_step": 2, "did_update": True, "loss": 0.8, "policy_loss": 0.4, "policy_lag_updates": 0, "vllm_policy_version": 1, "vllm_post_update_policy_version": 2, "vllm_policy_fingerprint": "b", "vllm_post_update_policy_fingerprint": "c", "rollout_request_id": "u1-r2", "rollout_prompt_fingerprint": "p2", "rollout_sampling_fingerprint": "s", "vllm_engine_policy_verified": True, "vllm_export_transaction_committed": True},
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


def test_sync_artifact_validator_checks_active_manifest_pair(tmp_path):
    raw = tmp_path / "raw-policy-u2-abc"
    export = tmp_path / "hf-policy-u2-abc"
    raw.mkdir()
    export.mkdir()
    manifest = {
        "complete": True,
        "policy_version": 2,
        "policy_fingerprint": "abcdef",
        "raw_checkpoint_name": raw.name,
        "export_name": export.name,
        "roundtrip_validated": True,
    }
    (export / "fitmotn_vllm_sync_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    state = {
        "policy_version": 2,
        "policy_fingerprint": "abcdef",
        "export_dir": str(export),
        "raw_checkpoint_dir": str(raw),
    }
    (tmp_path / "fitmotn_vllm_sync_state.json").write_text(json.dumps(state), encoding="utf-8")

    result = SYNC_MODULE.validate_sync_root(tmp_path, min_policy_version=2)

    assert result["ok"] is True
    assert result["committed_export_count"] == 1


def test_sync_artifact_validator_rejects_incomplete_transaction(tmp_path):
    (tmp_path / ".hf-policy-u1-abc.tmp-deadbeef").mkdir()
    result = SYNC_MODULE.validate_sync_root(tmp_path)
    assert result["ok"] is False
    assert any("incomplete transaction" in error for error in result["errors"])


def test_sync_artifact_validator_rejects_orphan_raw_checkpoint(tmp_path):
    (tmp_path / "raw-policy-u1-orphan").mkdir()
    result = SYNC_MODULE.validate_sync_root(tmp_path)
    assert result["ok"] is False
    assert any("orphan raw" in error for error in result["errors"])
