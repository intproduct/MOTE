from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from ..checkpointing import validate_checkpoint
from ..train.stages import build_stage_plan
from ..rl.device_topology import (
    model_tp_compatibility,
    rollout_topology_config,
    topology_overlap,
)


def _directory_bytes(path: Path) -> int:
    total = 0
    if not path.is_dir():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += int(item.stat().st_size)
        except OSError:
            continue
    return total


def inspect_config(cfg) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    plan = build_stage_plan(cfg.train)
    total_updates = int(plan.total_updates)
    save_every = int(cfg.train.save_every_updates)
    periodic_steps = set(range(save_every, total_updates + 1, save_every))
    if save_every > total_updates:
        warnings.append(
            f"train.save_every_updates={save_every} exceeds total updates={total_updates}; "
            "only final_model will be saved unless a stage-boundary save is triggered"
        )
    boundary_steps = {
        int(stage.end_update)
        for stage in plan.stages[:-1]
    } if bool(cfg.train.save_on_stage_transition) else set()
    generated_steps = periodic_steps | boundary_steps
    ordered_steps = sorted(generated_steps)
    keep_last_n = int(cfg.train.checkpoint_keep_last_n)
    retained_steps = set(ordered_steps[-keep_last_n:]) if keep_last_n > 0 else set()
    keep_every_n = int(cfg.train.checkpoint_keep_every_n)
    if keep_every_n > 0:
        retained_steps.update(step for step in ordered_steps if step % keep_every_n == 0)
    retained_steps.update(boundary_steps)
    estimated_checkpoint_count = len(retained_steps) + 1
    peak_checkpoint_count = estimated_checkpoint_count + 1
    model_path = Path(str(cfg.model.model_path)).expanduser()
    model_bytes = _directory_bytes(model_path)
    estimated_disk_bytes = model_bytes * peak_checkpoint_count if model_bytes > 0 else None
    output_root = Path(str(cfg.output.root_dir)).expanduser()
    probe = output_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free_bytes = int(shutil.disk_usage(probe).free)
    except OSError:
        free_bytes = None
    if not os.access(probe, os.W_OK):
        errors.append(f"output filesystem is not writable at nearest existing path: {probe}")
    if estimated_disk_bytes is not None and free_bytes is not None and estimated_disk_bytes > free_bytes:
        errors.append(
            f"estimated SFT checkpoint bytes {estimated_disk_bytes} exceed free disk bytes {free_bytes}"
        )
    elif estimated_disk_bytes is not None and free_bytes is not None and estimated_disk_bytes > free_bytes * 0.8:
        warnings.append("estimated SFT checkpoints may consume more than 80% of currently free disk")

    train_exact = getattr(cfg.train, "resume_checkpoint_from", None)
    rl_exact = getattr(cfg.rl, "resume_checkpoint_from", None)
    for kind, path in (("sft", train_exact), ("rl", rl_exact)):
        if path:
            result = validate_checkpoint(path, require_exact_resume=kind)
            if not result["ok"]:
                errors.append(f"{kind} exact-resume checkpoint is invalid: {result['errors']}")

    rl_generated_count = 0
    rl_retained_count = 0
    rl_peak_count = 0
    rl_estimated_disk_bytes = None
    rl_resource_topology = None
    if bool(getattr(cfg.rl, "enabled", False)):
        rl_updates = int(cfg.rl.max_steps)
        rl_save_every = int(cfg.rl.save_every_updates)
        if rl_save_every > 0 and rl_save_every > rl_updates:
            warnings.append(
                f"rl.save_every_updates={rl_save_every} exceeds rl.max_steps={rl_updates}; "
                "only the RL final_model will be saved"
            )
        rl_steps = set(range(rl_save_every, rl_updates + 1, rl_save_every)) if rl_save_every > 0 else set()
        rl_generated_count = len(rl_steps)
        rl_keep_last_n = int(cfg.rl.checkpoint_keep_last_n)
        ordered_rl_steps = sorted(rl_steps)
        retained_rl_steps = set(ordered_rl_steps[-rl_keep_last_n:]) if rl_keep_last_n > 0 else set()
        rl_keep_every_n = int(cfg.rl.checkpoint_keep_every_n)
        if rl_keep_every_n > 0:
            retained_rl_steps.update(step for step in ordered_rl_steps if step % rl_keep_every_n == 0)
        rl_retained_count = len(retained_rl_steps) + 1
        rl_peak_count = rl_retained_count + 1
        rl_estimated_disk_bytes = model_bytes * rl_peak_count if model_bytes > 0 else None
        if rl_estimated_disk_bytes is not None and free_bytes is not None and rl_estimated_disk_bytes > free_bytes:
            errors.append(
                f"estimated RL checkpoint bytes {rl_estimated_disk_bytes} exceed free disk bytes {free_bytes}"
            )
        elif rl_estimated_disk_bytes is not None and free_bytes is not None and rl_estimated_disk_bytes > free_bytes * 0.8:
            warnings.append("estimated RL checkpoints may consume more than 80% of currently free disk")
        if str(getattr(cfg.rl, "rollout_backend", "hf")) == "vllm":
            train_device = str(getattr(cfg.model, "device", ""))
            rollout_device = str(getattr(cfg.rl, "vllm_device", "") or "")
            rl_resource_topology = rollout_topology_config(cfg.rl)
            actor_specs = list(rl_resource_topology["actors"])
            expanded_rows = int(getattr(cfg.rl, "batch_size", 1)) * int(getattr(cfg.rl, "group_size", 1))
            if len(actor_specs) > expanded_rows:
                warnings.append(
                    f"configured {len(actor_specs)} rollout actors but each micro-step has only "
                    f"batch_size*group_size={expanded_rows} rollout rows; some actors will be idle"
                )
            isolated_actor_devices = [device for actor in actor_specs for device in actor["cuda_visible_devices"]]
            if not isolated_actor_devices and rollout_device and rollout_device == train_device:
                warnings.append("trainer and vLLM are configured on the same device; check memory headroom")
            for actor in actor_specs:
                actor_visible = list(actor["cuda_visible_devices"])
                overlap = topology_overlap(train_device, actor_visible)
                actor["trainer_actor_overlap"] = overlap
                if overlap["overlap"]:
                    errors.append(
                        "trainer and isolated vLLM actor CUDA device sets overlap: "
                        f"actor={actor['name']}, details={overlap}"
                    )
                tp_report = model_tp_compatibility(
                    cfg.model.model_path,
                    int(actor["tensor_parallel_size"]),
                )
                actor["model_tp_compatibility"] = tp_report
                if tp_report.get("ok") is False:
                    errors.append(
                        f"model dimensions are incompatible with actor {actor['name']!r} TP size: "
                        f"{tp_report.get('incompatible_dimensions')}"
                    )
                elif int(actor["tensor_parallel_size"]) > 1 and not tp_report.get("checked"):
                    warnings.append(
                        f"could not statically inspect model config for actor {actor['name']!r} "
                        "tensor-parallel divisibility; target GPU preflight remains required"
                    )
            if not isolated_actor_devices and str(getattr(cfg.rl, "vllm_execution_mode", "in_process")) == "subprocess":
                warnings.append(
                    "subprocess vLLM has no rl.vllm_actor_cuda_visible_devices isolation; "
                    "use an explicit one-device list even for TP=1"
                )
            if bool(getattr(cfg.rl, "vllm_fallback_to_hf", False)):
                warnings.append("rl.vllm_fallback_to_hf=true can hide vLLM failures in formal experiments")
            if rl_exact:
                warnings.append(
                    "RL exact resume with vLLM restores FitMoTN/optimizer/data/Python/Torch state, "
                    "but vLLM engine-internal sampling state is not guaranteed bitwise identical"
                )
        if str(getattr(cfg.rl, "trainable_mode", "patch_only")) == "all":
            errors.append(
                "rl.trainable_mode='all' is unsafe with patch_state_only_v2 checkpoints; "
                "dense updates are not guaranteed to resume or export. Use patch_only until checkpoint v3."
            )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "sft": {
            "total_updates": total_updates,
            "stage_count": len(plan.stages),
            "periodic_checkpoint_count": len(periodic_steps),
            "stage_boundary_checkpoint_count": len(boundary_steps),
            "generated_checkpoint_count_before_retention": len(generated_steps),
            "estimated_retained_checkpoint_count_including_final": estimated_checkpoint_count,
            "estimated_peak_checkpoint_count_during_atomic_save": peak_checkpoint_count,
            "checkpoint_keep_last_n": int(cfg.train.checkpoint_keep_last_n),
            "estimated_checkpoint_disk_bytes": estimated_disk_bytes,
        },
        "rl": {
            "enabled": bool(getattr(cfg.rl, "enabled", False)),
            "max_steps": int(getattr(cfg.rl, "max_steps", 0)),
            "checkpoint_keep_last_n": int(getattr(cfg.rl, "checkpoint_keep_last_n", 3)),
            "generated_checkpoint_count_before_retention": rl_generated_count,
            "estimated_retained_checkpoint_count_including_final": rl_retained_count,
            "estimated_peak_checkpoint_count_during_atomic_save": rl_peak_count,
            "estimated_checkpoint_disk_bytes": rl_estimated_disk_bytes,
            "resource_topology": rl_resource_topology,
        },
        "disk": {
            "probe_path": str(probe),
            "free_bytes": free_bytes,
        },
    }
