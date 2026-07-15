from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


ROLLOUT_ACTOR_ALLOWED_KEYS = {
    "name",
    "cuda_visible_devices",
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "seed_offset",
}


def normalize_visible_devices(value: Any) -> list[str]:
    """Normalize an actor CUDA visibility declaration without interpreting IDs."""
    if value is None:
        return []
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        raise ValueError("rl.vllm_actor_cuda_visible_devices must be a list or comma-separated string")
    normalized = [str(item).strip() for item in values if str(item).strip()]
    if len(normalized) != len(set(normalized)):
        raise ValueError("rl.vllm_actor_cuda_visible_devices contains duplicate device identifiers")
    return normalized


def normalize_rollout_actor_configs(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("rl.vllm_rollout_actors must be a list of actor objects")
    normalized = []
    names = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"rl.vllm_rollout_actors[{index}] must be an object")
        unknown = sorted(set(raw) - ROLLOUT_ACTOR_ALLOWED_KEYS)
        if unknown:
            raise ValueError(f"unknown keys in rl.vllm_rollout_actors[{index}]: {unknown}")
        name = str(raw.get("name") or f"actor_{index}").strip()
        if not name or name in names:
            raise ValueError(f"rl.vllm_rollout_actors contains an empty or duplicate actor name: {name!r}")
        names.add(name)
        devices = normalize_visible_devices(raw.get("cuda_visible_devices"))
        if not devices:
            raise ValueError(f"rl.vllm_rollout_actors[{index}].cuda_visible_devices cannot be empty")
        tp_size = int(raw.get("tensor_parallel_size", len(devices)))
        if tp_size < 1 or len(devices) != tp_size:
            raise ValueError(
                f"actor {name!r} requires exactly tensor_parallel_size={tp_size} CUDA devices; got {devices}"
            )
        actor = {
            "name": name,
            "cuda_visible_devices": devices,
            "tensor_parallel_size": tp_size,
        }
        for key in ("gpu_memory_utilization", "max_model_len", "max_num_seqs", "seed_offset"):
            if key in raw and raw[key] is not None:
                actor[key] = float(raw[key]) if key == "gpu_memory_utilization" else int(raw[key])
        if "gpu_memory_utilization" in actor and not (0.0 < actor["gpu_memory_utilization"] <= 1.0):
            raise ValueError(f"actor {name!r} gpu_memory_utilization must satisfy 0 < value <= 1")
        for key in ("max_model_len", "max_num_seqs"):
            if key in actor and actor[key] <= 0:
                raise ValueError(f"actor {name!r} {key} must be > 0 when set")
        normalized.append(actor)
    return normalized


def parent_visible_devices(environ: dict[str, str] | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    raw = str(env.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    return [part.strip() for part in raw.split(",") if part.strip()]


def logical_cuda_index(device: str | None) -> int | None:
    value = str(device or "").strip().lower()
    if value == "cuda":
        return 0
    if not value.startswith("cuda:"):
        return None
    suffix = value.split(":", 1)[1]
    return int(suffix) if suffix.isdigit() else None


def physical_device_token(device: str | None, parent_visible: Sequence[str] | None = None) -> str | None:
    index = logical_cuda_index(device)
    if index is None:
        return None
    visible = list(parent_visible if parent_visible is not None else parent_visible_devices())
    if visible:
        return visible[index] if 0 <= index < len(visible) else None
    return str(index)


def resolve_actor_visible_devices(
    configured: Iterable[str],
    parent_visible: Sequence[str] | None = None,
) -> list[str]:
    """Resolve actor selectors against a scheduler-provided parent mask.

    Exact tokens (including GPU/MIG UUIDs and physical IDs already present in
    the mask) are preserved. Other integer selectors are interpreted as
    logical indices in the trainer process CUDA namespace.
    """
    configured_tokens = [str(item) for item in configured]
    parent = list(parent_visible if parent_visible is not None else parent_visible_devices())
    resolved = []
    for token in configured_tokens:
        if not parent or token in parent:
            resolved.append(token)
            continue
        if token.isdigit() and 0 <= int(token) < len(parent):
            resolved.append(parent[int(token)])
            continue
        raise ValueError(
            "rl.vllm_actor_cuda_visible_devices selects a device outside the parent CUDA mask: "
            f"selector={token!r}, parent={parent}"
        )
    if len(resolved) != len(set(resolved)):
        raise ValueError(
            "rl.vllm_actor_cuda_visible_devices resolves to duplicate devices under the parent CUDA mask: "
            f"configured={configured_tokens}, parent={parent}, resolved={resolved}"
        )
    return resolved


def actor_topology_config(rl_cfg, *, environ: dict[str, str] | None = None) -> dict[str, Any]:
    configured = normalize_visible_devices(getattr(rl_cfg, "vllm_actor_cuda_visible_devices", None))
    parent = parent_visible_devices(environ)
    visible = resolve_actor_visible_devices(configured, parent)
    tp_size = int(getattr(rl_cfg, "vllm_tensor_parallel_size", 1))
    device = getattr(rl_cfg, "vllm_device", None)
    return {
        "execution_mode": str(getattr(rl_cfg, "vllm_execution_mode", "in_process") or "in_process"),
        "actor_cuda_visible_devices": visible,
        "actor_cuda_visible_devices_configured": configured,
        "actor_cuda_visible_devices_env": ",".join(visible) if visible else None,
        "actor_device": None if device is None else str(device),
        "tensor_parallel_size": tp_size,
        "parent_cuda_visible_devices": parent,
    }


def rollout_actor_specs(rl_cfg, *, environ: dict[str, str] | None = None) -> list[dict[str, Any]]:
    configured_actors = normalize_rollout_actor_configs(getattr(rl_cfg, "vllm_rollout_actors", None))
    parent = parent_visible_devices(environ)
    if not configured_actors:
        topology = actor_topology_config(rl_cfg, environ=environ)
        return [
            {
                "name": "actor_0",
                "cuda_visible_devices_configured": topology["actor_cuda_visible_devices_configured"],
                "cuda_visible_devices": topology["actor_cuda_visible_devices"],
                "tensor_parallel_size": topology["tensor_parallel_size"],
                "gpu_memory_utilization": float(getattr(rl_cfg, "vllm_gpu_memory_utilization", 0.85)),
                "max_model_len": int(getattr(rl_cfg, "vllm_max_model_len", 0)),
                "max_num_seqs": int(getattr(rl_cfg, "vllm_max_num_seqs", 0)),
                "engine_seed": int(getattr(rl_cfg, "seed", 0) or 0),
            }
        ]
    specs = []
    used_devices: dict[str, str] = {}
    base_seed = int(getattr(rl_cfg, "seed", 0) or 0)
    for actor_index, actor in enumerate(configured_actors):
        configured_devices = list(actor["cuda_visible_devices"])
        resolved = resolve_actor_visible_devices(configured_devices, parent)
        for token in resolved:
            if token in used_devices:
                raise ValueError(
                    "vLLM rollout actor device sets overlap: "
                    f"device={token!r}, actors={used_devices[token]!r},{actor['name']!r}"
                )
            used_devices[token] = actor["name"]
        specs.append(
            {
                **actor,
                "cuda_visible_devices_configured": configured_devices,
                "cuda_visible_devices": resolved,
                "gpu_memory_utilization": float(
                    actor.get("gpu_memory_utilization", getattr(rl_cfg, "vllm_gpu_memory_utilization", 0.85))
                ),
                "max_model_len": int(actor.get("max_model_len", getattr(rl_cfg, "vllm_max_model_len", 0))),
                "max_num_seqs": int(actor.get("max_num_seqs", getattr(rl_cfg, "vllm_max_num_seqs", 0))),
                "engine_seed": int(base_seed + actor_index * 1_000_003 + int(actor.get("seed_offset", 0))),
            }
        )
    return specs


def rollout_topology_config(rl_cfg, *, environ: dict[str, str] | None = None) -> dict[str, Any]:
    specs = rollout_actor_specs(rl_cfg, environ=environ)
    return {
        "execution_mode": str(getattr(rl_cfg, "vllm_execution_mode", "in_process") or "in_process"),
        "actor_count": len(specs),
        "actors": specs,
        "total_rollout_gpus": sum(len(spec["cuda_visible_devices"]) for spec in specs),
        "parent_cuda_visible_devices": parent_visible_devices(environ),
    }


def validate_actor_topology(rl_cfg) -> None:
    topology = actor_topology_config(rl_cfg)
    visible = topology["actor_cuda_visible_devices"]
    execution_mode = topology["execution_mode"]
    tp_size = topology["tensor_parallel_size"]
    configured_actors = normalize_rollout_actor_configs(getattr(rl_cfg, "vllm_rollout_actors", None))
    if configured_actors:
        if execution_mode != "subprocess":
            raise ValueError("rl.vllm_rollout_actors requires rl.vllm_execution_mode='subprocess'")
        if visible:
            raise ValueError(
                "rl.vllm_rollout_actors and legacy rl.vllm_actor_cuda_visible_devices are mutually exclusive"
            )
        rollout_actor_specs(rl_cfg)
        return
    if visible and execution_mode != "subprocess":
        raise ValueError(
            "rl.vllm_actor_cuda_visible_devices requires rl.vllm_execution_mode='subprocess'"
        )
    if visible:
        if str(getattr(rl_cfg, "vllm_actor_start_method", "spawn")) != "spawn":
            raise ValueError(
                "isolated vLLM actor CUDA visibility requires rl.vllm_actor_start_method='spawn'"
            )
        if len(visible) != tp_size:
            raise ValueError(
                "rl.vllm_actor_cuda_visible_devices must contain exactly "
                "rl.vllm_tensor_parallel_size devices; "
                f"got devices={visible}, tensor_parallel_size={tp_size}"
            )
        local_index = logical_cuda_index(getattr(rl_cfg, "vllm_device", None) or "cuda:0")
        if local_index != 0:
            raise ValueError(
                "with isolated actor devices, rl.vllm_device must be 'cuda' or 'cuda:0' because "
                "CUDA devices are remapped inside the actor"
            )
    elif execution_mode == "subprocess" and tp_size > 1:
        raise ValueError(
            "subprocess vLLM tensor parallelism requires rl.vllm_actor_cuda_visible_devices so "
            "trainer and rollout workers cannot collide"
        )


def model_tp_compatibility(model_path: str | Path, tp_size: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "checked": False,
        "ok": None,
        "tensor_parallel_size": int(tp_size),
        "model_config_path": None,
        "dimensions": {},
        "incompatible_dimensions": {},
    }
    if int(tp_size) <= 1:
        report.update({"checked": True, "ok": True})
        return report
    config_path = Path(str(model_path)).expanduser() / "config.json"
    report["model_config_path"] = str(config_path)
    if not config_path.is_file():
        return report
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report["error"] = str(exc)
        return report
    dimensions = {}
    for key in ("hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size"):
        value = payload.get(key)
        if isinstance(value, int) and value > 0:
            dimensions[key] = int(value)
    incompatible = {key: value for key, value in dimensions.items() if value % int(tp_size) != 0}
    report.update(
        {
            "checked": True,
            "ok": not incompatible,
            "dimensions": dimensions,
            "incompatible_dimensions": incompatible,
        }
    )
    return report


def topology_overlap(
    trainer_device: str | None,
    actor_visible_devices: Iterable[str],
    *,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    parent = parent_visible_devices(environ)
    trainer_token = physical_device_token(trainer_device, parent)
    actor_tokens = [str(item) for item in actor_visible_devices]
    return {
        "trainer_device": trainer_device,
        "trainer_physical_token": trainer_token,
        "actor_physical_tokens": actor_tokens,
        "overlap": bool(trainer_token is not None and trainer_token in actor_tokens),
        "parent_cuda_visible_devices": parent,
    }
