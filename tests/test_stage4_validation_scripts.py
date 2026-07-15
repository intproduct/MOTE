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
STAGE5B_SPEC = importlib.util.spec_from_file_location(
    "validate_stage5b_multigpu", ROOT / "scripts" / "validate_stage5b_multigpu.py"
)
STAGE5B_MODULE = importlib.util.module_from_spec(STAGE5B_SPEC)
assert STAGE5B_SPEC.loader is not None
STAGE5B_SPEC.loader.exec_module(STAGE5B_MODULE)
STAGE5C_SPEC = importlib.util.spec_from_file_location(
    "validate_stage5c_multiactor", ROOT / "scripts" / "validate_stage5c_multiactor.py"
)
STAGE5C_MODULE = importlib.util.module_from_spec(STAGE5C_SPEC)
assert STAGE5C_SPEC.loader is not None
STAGE5C_SPEC.loader.exec_module(STAGE5C_MODULE)


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


def test_stage5b_validator_accepts_isolated_tp2_evidence(tmp_path, monkeypatch):
    topology = {
        "execution_mode": "subprocess",
        "actors": [{"name": "actor_0", "cuda_visible_devices": ["1", "2"], "tensor_parallel_size": 2}],
    }
    monkeypatch.setattr(STAGE5B_MODULE, "load_config", lambda **kwargs: object())
    monkeypatch.setattr(
        STAGE5B_MODULE,
        "inspect_config",
        lambda cfg: {"ok": True, "errors": [], "warnings": [], "rl": {"resource_topology": topology}},
    )
    log = tmp_path / "rl_train.jsonl"
    actor_resources = {
        "engine": {
            "actor_resources": {
                "cuda_device_count": 2,
                "cuda_available": True,
                "cuda_visible_devices": "1,2",
                "process_id": 100,
                "process_group_id": 100,
            },
            "engine_topology": {"observed_tensor_parallel_size": 2, "tensor_parallel_verified": True},
        }
    }
    records = [
        {"kind": "run_start", "vllm_actor_resources": actor_resources},
        {"kind": "train", "update_step": 1, "policy_lag_updates": 0, "vllm_fallback_used": False, "vllm_engine_policy_verified": True},
        {"kind": "train", "update_step": 2, "policy_lag_updates": 0, "vllm_fallback_used": False, "vllm_engine_policy_verified": True},
    ]
    log.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")

    result = STAGE5B_MODULE.validate(tmp_path / "config.json", log, min_updates=2)
    assert result["ok"] is True


def test_stage5b_validator_rejects_actor_count_and_policy_lag(tmp_path, monkeypatch):
    topology = {
        "execution_mode": "subprocess",
        "actors": [{"name": "actor_0", "cuda_visible_devices": ["1", "2"], "tensor_parallel_size": 2}],
    }
    monkeypatch.setattr(STAGE5B_MODULE, "load_config", lambda **kwargs: object())
    monkeypatch.setattr(
        STAGE5B_MODULE,
        "inspect_config",
        lambda cfg: {"ok": True, "errors": [], "warnings": [], "rl": {"resource_topology": topology}},
    )
    log = tmp_path / "rl_train.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "run_start",
                        "vllm_actor_resources": {
                            "engine": {
                                "actor_resources": {
                                    "cuda_device_count": 1,
                                    "cuda_available": True,
                                    "cuda_visible_devices": "1",
                                    "process_id": 100,
                                    "process_group_id": 100,
                                },
                                "engine_topology": {"observed_tensor_parallel_size": 2, "tensor_parallel_verified": True},
                            }
                        },
                    }
                ),
                json.dumps(
                    {"kind": "train", "update_step": 1, "policy_lag_updates": 1, "vllm_fallback_used": False, "vllm_engine_policy_verified": True}
                ),
            ]
        ),
        encoding="utf-8",
    )
    result = STAGE5B_MODULE.validate(tmp_path / "config.json", log, min_updates=1)
    assert result["ok"] is False
    assert any("observed 1 CUDA" in error for error in result["errors"])
    assert any("policy lag" in error for error in result["errors"])


def test_stage5c_validator_accepts_complete_multi_actor_dispatch(tmp_path, monkeypatch):
    actors = [
        {"name": "r0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1, "engine_seed": 1},
        {"name": "r1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1, "engine_seed": 2},
    ]
    topology = {"execution_mode": "subprocess", "actors": actors, "actor_count": 2}
    monkeypatch.setattr(STAGE5C_MODULE, "load_config", lambda **kwargs: object())
    monkeypatch.setattr(
        STAGE5C_MODULE,
        "inspect_config",
        lambda cfg: {"ok": True, "errors": [], "warnings": [], "rl": {"resource_topology": topology}},
    )

    def snapshot(device):
        return {
            "engine": {
                "actor_resources": {
                    "cuda_visible_devices": device,
                    "cuda_device_count": 1,
                    "cuda_available": True,
                    "process_id": 100 + int(device),
                    "process_group_id": 100 + int(device),
                },
                "engine_topology": {
                    "observed_tensor_parallel_size": 1,
                    "tensor_parallel_verified": True,
                },
            },
            "policy_verified": True,
        }

    descriptor = {"policy_version": 0, "policy_fingerprint": "fp0", "export_dir": "export0"}
    records = [
        {
            "kind": "run_start",
            "vllm_actor_resources": {"r0": snapshot("1"), "r1": snapshot("2")},
            "vllm_all_actors_policy_verified": True,
        },
        {
            "kind": "train",
            "update_step": 1,
            "policy_lag_updates": 0,
            "vllm_fallback_used": False,
            "vllm_engine_policy_verified": True,
            "vllm_all_actors_policy_verified": True,
            "vllm_rollout_actor_count": 2,
            "vllm_policy_version": 0,
            "vllm_policy_fingerprint": "fp0",
            "vllm_export_dir": "export0",
            "vllm_actor_dispatch": {
                "row_count": 4,
                "group_size": 4,
                "active_actor_count": 2,
                "actors": {
                    "r0": {"row_indices": [0, 2], "policy_descriptor": descriptor, "engine_seed": 1},
                    "r1": {"row_indices": [1, 3], "policy_descriptor": descriptor, "engine_seed": 2},
                },
            },
        },
    ]
    log = tmp_path / "rl_train.jsonl"
    log.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    result = STAGE5C_MODULE.validate(tmp_path / "config.json", log, min_updates=1)
    assert result["ok"] is True, result
