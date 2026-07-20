from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from validate_stage5e_patch_sync import validate


def _config(path: Path) -> Path:
    value = {
        "runtime": {"dev_mode": True},
        "rl": {
            "rollout_backend": "vllm",
            "trainable_mode": "patch_only",
            "vllm_execution_mode": "subprocess",
            "vllm_sync_strategy": "weight_transfer_nccl",
            "vllm_native_transfer_required_level": "update_only",
            "vllm_weight_transfer_scope": "trainable_patch",
            "vllm_rollout_actors": [
                {"name": "a0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
                {"name": "a1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1},
            ],
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _actor_result(pid: int, fingerprint: str, tensor_count: int) -> dict:
    return {
        "committed": True,
        "receive": {
            "engine_process_id": pid,
            "rollout_facing_validation": {"ok": True},
            "received_weight_update_contract": {
                "transfer_scope": "trainable_patch",
                "transfer_plan_fingerprint": fingerprint,
                "tensor_count": tensor_count,
            },
        },
    }


def _run(path: Path, *, payload_ratio: float = 0.02, teardown_ok: bool = True) -> Path:
    fingerprint = "plan-1"
    tensor_count = 42
    rows = [
        {
            "kind": "run_start",
            "vllm_weight_transfer_bootstrap": True,
            "patch_transfer_plan": {"transfer_scope": "trainable_patch"},
        }
    ]
    for update in (1, 2):
        rows.append(
            {
                "kind": "train",
                "update_step": update,
                "vllm_policy_version": update,
                "policy_lag_updates": 0,
                "vllm_weight_transfer_native_sync": True,
                "vllm_weight_transfer_scope": "trainable_patch",
                "weight_transfer_adapter": "nccl_update_only_subprocess",
                "weight_transfer_plan_fingerprint": fingerprint,
                "weight_transfer_tensor_count": tensor_count,
                "weight_transfer_bytes": 200,
                "full_policy_bytes": 10000,
                "weight_transfer_payload_ratio": payload_ratio,
                "frozen_parameter_drift": False,
                "weight_transfer_commit_barrier": True,
                "weight_transfer_actor_count": 2,
                "weight_transfer_actor_results": {
                    "a0": _actor_result(100, fingerprint, tensor_count),
                    "a1": _actor_result(200, fingerprint, tensor_count),
                },
                "vllm_sync_sec": 1.0 + update,
                "vllm_engine_rebuild_sec": 0.0,
                "vllm_fallback_used": False,
            }
        )
    rows.append(
        {
            "kind": "run_end",
            "update_step": 2,
            "rollout_teardown": {"ok": teardown_ok, "backend": "vllm"},
        }
    )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_stage5e_validator_accepts_patch_only_native_updates(tmp_path):
    export_root = tmp_path / "vllm_sync"
    (export_root / "hf-policy-u0-x").mkdir(parents=True)
    (export_root / "raw-policy-u0-x").mkdir()
    result = validate(
        config_path=_config(tmp_path / "config.json"),
        run_path=_run(tmp_path / "rl_train.jsonl"),
        export_root=export_root,
        min_updates=2,
        max_payload_ratio=0.10,
        max_sync_p95_sec=10.0,
    )
    assert result["ok"] is True, result
    assert result["run"]["payload_ratio_mean"] == 0.02


def test_stage5e_validator_rejects_full_sized_payload(tmp_path):
    result = validate(
        config_path=_config(tmp_path / "config.json"),
        run_path=_run(tmp_path / "rl_train.jsonl", payload_ratio=1.0),
        export_root=None,
        min_updates=2,
        max_payload_ratio=0.10,
        max_sync_p95_sec=None,
    )
    assert result["ok"] is False
    assert any("payload ratio" in error for error in result["errors"])


def test_stage5e_validator_rejects_failed_rollout_teardown(tmp_path):
    result = validate(
        config_path=_config(tmp_path / "config.json"),
        run_path=_run(tmp_path / "rl_train.jsonl", teardown_ok=False),
        export_root=None,
        min_updates=2,
        max_payload_ratio=0.10,
        max_sync_p95_sec=None,
    )
    assert result["ok"] is False
    assert any("clean rollout teardown" in error for error in result["errors"])
