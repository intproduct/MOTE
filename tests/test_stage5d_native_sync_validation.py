from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from validate_stage5d_native_sync import validate


def _config(path: Path) -> Path:
    value = {
        "runtime": {"dev_mode": True},
        "rl": {
            "rollout_backend": "vllm",
            "vllm_execution_mode": "subprocess",
            "vllm_sync_strategy": "weight_transfer_nccl",
            "vllm_native_transfer_required_level": "update_only",
            "vllm_weight_transfer_master_port": 29580,
            "vllm_rollout_actors": [
                {"name": "a0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
                {"name": "a1", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1},
            ],
            "vllm_weight_transfer_fallback_to_export_reload": False,
            "vllm_fallback_to_hf": False,
            "allow_stale_vllm_policy": False,
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _actor_result(pid: int) -> dict:
    return {
        "committed": True,
        "receive": {
            "engine_process_id": pid,
            "rollout_facing_validation": {"ok": True},
        },
    }


def _run(path: Path, *, changed_pid: bool = False) -> Path:
    rows = [
        {
            "kind": "run_start",
            "vllm_weight_transfer_bootstrap": True,
            "vllm_weight_transfer_native_sync": False,
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
                "weight_transfer_adapter": "nccl_update_only_subprocess",
                "weight_transfer_commit_barrier": True,
                "weight_transfer_actor_count": 2,
                "weight_transfer_actor_results": {
                    "a0": _actor_result(100 if not changed_pid or update == 1 else 101),
                    "a1": _actor_result(200),
                },
                "vllm_sync_sec": 10.0 + update,
                "vllm_engine_rebuild_sec": 0.0,
                "vllm_fallback_used": False,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_stage5d_validator_accepts_persistent_native_updates(tmp_path):
    export_root = tmp_path / "vllm_sync"
    (export_root / "hf-policy-u0-x").mkdir(parents=True)
    (export_root / "raw-policy-u0-x").mkdir()
    result = validate(
        config_path=_config(tmp_path / "config.json"),
        run_path=_run(tmp_path / "rl_train.jsonl"),
        export_root=export_root,
        min_updates=2,
        max_sync_p95_sec=60.0,
    )
    assert result["ok"] is True, result
    assert result["run"]["actor_pids"] == {"a0": [100], "a1": [200]}


def test_stage5d_validator_rejects_engine_rebuild_hidden_as_pid_change(tmp_path):
    result = validate(
        config_path=_config(tmp_path / "config.json"),
        run_path=_run(tmp_path / "rl_train.jsonl", changed_pid=True),
        export_root=None,
        min_updates=2,
        max_sync_p95_sec=None,
    )
    assert result["ok"] is False
    assert any("PID changed" in error for error in result["errors"])
