from __future__ import annotations

import json
from pathlib import Path

import pytest

from MOTE.config.loader import load_config_from_json
from MOTE.config.defaults import make_default_config
from MOTE.diagnostics.config_doctor import inspect_config
from MOTE.rl.device_topology import (
    actor_topology_config,
    model_tp_compatibility,
    normalize_visible_devices,
    resolve_actor_visible_devices,
    rollout_actor_specs,
    rollout_topology_config,
    topology_overlap,
)


def _write_config(tmp_path: Path, rl: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"runtime": {"dev_mode": True}, "rl": rl}), encoding="utf-8")
    return path


def test_normalize_actor_visible_devices():
    assert normalize_visible_devices("1, 2") == ["1", "2"]
    assert normalize_visible_devices(["GPU-a", "GPU-b"]) == ["GPU-a", "GPU-b"]
    with pytest.raises(ValueError, match="duplicate"):
        normalize_visible_devices(["1", "1"])


def test_subprocess_tp2_requires_explicit_isolation(tmp_path):
    path = _write_config(
        tmp_path,
        {"vllm_execution_mode": "subprocess", "vllm_tensor_parallel_size": 2},
    )
    with pytest.raises(ValueError, match="requires rl.vllm_actor_cuda_visible_devices"):
        load_config_from_json(path)


def test_isolated_actor_requires_exact_tp_device_count(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "vllm_execution_mode": "subprocess",
            "vllm_tensor_parallel_size": 2,
            "vllm_actor_cuda_visible_devices": ["1"],
        },
    )
    with pytest.raises(ValueError, match="exactly"):
        load_config_from_json(path)


def test_isolated_actor_remaps_device_zero(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "vllm_execution_mode": "subprocess",
            "vllm_tensor_parallel_size": 2,
            "vllm_actor_cuda_visible_devices": ["1", "2"],
        },
    )
    cfg = load_config_from_json(path)
    assert cfg.rl.vllm_device == "cuda:0"
    assert cfg.rl.vllm_actor_cuda_visible_devices == ["1", "2"]


def test_loader_accepts_logical_actor_selectors_under_scheduler_mask(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6")
    path = _write_config(
        tmp_path,
        {
            "vllm_execution_mode": "subprocess",
            "vllm_tensor_parallel_size": 2,
            "vllm_actor_cuda_visible_devices": ["1", "2"],
        },
    )
    cfg = load_config_from_json(path)
    topology = actor_topology_config(cfg.rl)
    assert topology["actor_cuda_visible_devices"] == ["5", "6"]
    assert topology["actor_cuda_visible_devices_configured"] == ["1", "2"]


def test_isolated_actor_rejects_nonzero_local_device(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "vllm_execution_mode": "subprocess",
            "vllm_tensor_parallel_size": 1,
            "vllm_actor_cuda_visible_devices": ["2"],
            "vllm_device": "cuda:1",
        },
    )
    with pytest.raises(ValueError, match="remapped"):
        load_config_from_json(path)


def test_topology_overlap_maps_parent_logical_device():
    report = topology_overlap(
        "cuda:0",
        ["GPU-a", "GPU-b"],
        environ={"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-c"},
    )
    assert report["trainer_physical_token"] == "GPU-a"
    assert report["overlap"] is True


def test_actor_selectors_resolve_against_scheduler_mask():
    assert resolve_actor_visible_devices(["1", "2"], ["4", "5", "6"]) == ["5", "6"]
    assert resolve_actor_visible_devices(["GPU-b"], ["GPU-a", "GPU-b"]) == ["GPU-b"]
    with pytest.raises(ValueError, match="outside the parent CUDA mask"):
        resolve_actor_visible_devices(["3"], ["0", "1"])


def test_model_tp_compatibility_reads_local_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 11008,
            }
        ),
        encoding="utf-8",
    )
    assert model_tp_compatibility(tmp_path, 2)["ok"] is True
    report = model_tp_compatibility(tmp_path, 3)
    assert report["ok"] is False
    assert "hidden_size" in report["incompatible_dimensions"]


def test_doctor_rejects_trainer_actor_overlap(tmp_path, monkeypatch):
    cfg = make_default_config()
    cfg.train.epochs = 0
    cfg.train.steps = 1
    cfg.rl.enabled = True
    cfg.rl.max_steps = 1
    cfg.rl.rollout_backend = "vllm"
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.vllm_actor_cuda_visible_devices = ["4"]
    cfg.rl.vllm_tensor_parallel_size = 1
    cfg.model.device = "cuda:0"
    cfg.model.model_path = str(tmp_path / "model")
    cfg.output.root_dir = str(tmp_path / "runs")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    result = inspect_config(cfg)
    assert result["ok"] is False
    assert any("device sets overlap" in error for error in result["errors"])


def test_doctor_rejects_dense_all_mode(tmp_path):
    cfg = make_default_config()
    cfg.train.epochs = 0
    cfg.train.steps = 1
    cfg.rl.enabled = True
    cfg.rl.max_steps = 1
    cfg.rl.trainable_mode = "all"
    cfg.model.model_path = str(tmp_path / "model")
    cfg.output.root_dir = str(tmp_path / "runs")
    result = inspect_config(cfg)
    assert result["ok"] is False
    assert any("patch_state_only_v2" in error for error in result["errors"])


def test_stage5c_actor_specs_allow_mixed_tp_and_reject_overlap():
    cfg = make_default_config()
    cfg.rl.vllm_execution_mode = "subprocess"
    cfg.rl.vllm_rollout_actors = [
        {"name": "r0", "cuda_visible_devices": ["1"], "tensor_parallel_size": 1},
        {"name": "r1", "cuda_visible_devices": ["2", "3"], "tensor_parallel_size": 2},
    ]
    specs = rollout_actor_specs(cfg.rl, environ={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    assert [spec["tensor_parallel_size"] for spec in specs] == [1, 2]
    assert specs[0]["engine_seed"] != specs[1]["engine_seed"]
    topology = rollout_topology_config(cfg.rl, environ={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    assert topology["actor_count"] == 2
    assert topology["total_rollout_gpus"] == 3

    cfg.rl.vllm_rollout_actors[1]["cuda_visible_devices"] = ["1", "3"]
    with pytest.raises(ValueError, match="overlap"):
        rollout_actor_specs(cfg.rl, environ={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})


def test_stage5c_loader_rejects_legacy_and_actor_list_together(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "vllm_execution_mode": "subprocess",
            "vllm_actor_cuda_visible_devices": ["1"],
            "vllm_rollout_actors": [
                {"name": "r0", "cuda_visible_devices": ["2"], "tensor_parallel_size": 1}
            ],
        },
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config_from_json(path)
