from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import torch as tc
import torch.nn as nn
from .model import MOTNFFNLayer


PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


def resolve_layer_idxs(n_layers: int, mode: str) -> List[int]:
    mode = str(mode).lower()
    if mode == "all":
        return list(range(n_layers))
    if mode == "last_half":
        start = max(0, n_layers - max(1, n_layers // 2))
        return list(range(start, n_layers))
    if mode == "last_quarter":
        start = max(0, n_layers - max(1, n_layers // 4))
        return list(range(start, n_layers))
    if mode == "last_third":
        start = max(0, n_layers - max(1, n_layers // 3))
        return list(range(start, n_layers))
    raise ValueError(f"Unsupported layers_to_patch={mode}")


def patch_qwen_ffn_layers(model: nn.Module, layer_idxs: Iterable[int], motn_cfg: Dict[str, Any], device: tc.device, dtype=tc.float32, log=None) -> nn.Module:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("model does not have model.layers")
    layers = model.model.layers
    for idx in sorted(set(int(i) for i in layer_idxs)):
        layer = layers[idx]
        old_mlp = layer.mlp
        layer.mlp = MOTNFFNLayer(old_mlp, motn_cfg, device=device, dtype=dtype, log=log).to(device)
        layer.mlp.layer_idx = idx
        layer.mlp.fitmotn_block_layout = {}
        for name in PROJ_NAMES:
            layout = resolve_operator_block_layout(getattr(layer.mlp, name), motn_cfg)
            layer.mlp.fitmotn_block_layout[name] = layout
            setattr(getattr(layer.mlp, name), "fitmotn_block_layout", layout)
        layer.mlp.fitmotn_expert_warmup_state = {"enabled": False}
        if log is not None:
            log.info(f"[Patch] layer={idx} replace mlp: {type(old_mlp)} -> {type(layer.mlp)}")
    return model


def set_trainable_motn_only(model: nn.Module, log=None) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    patched = 0
    for module in model.modules():
        if isinstance(module, MOTNFFNLayer):
            patched += 1
            for p in module.parameters():
                p.requires_grad_(True)
    if log is not None:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info(f"[Freeze] patched_layers={patched} | trainable={trainable} / total={total}")


def iter_patched_motn_layers(model: nn.Module):
    items = []
    for module in model.modules():
        if isinstance(module, MOTNFFNLayer):
            items.append((int(getattr(module, "layer_idx", -1)), module))
    items.sort(key=lambda x: x[0])
    return items


def iter_trainable_params(model: nn.Module):
    for p in model.parameters():
        if p.requires_grad:
            yield p


def set_motn_temperature(model: nn.Module, temperature: float):
    for module in model.modules():
        if isinstance(module, MOTNFFNLayer):
            for proj in (module.gate_proj, module.up_proj, module.down_proj):
                try:
                    proj.core.gate.temperature = float(temperature)
                except Exception:
                    pass


def set_motn_gate_trainable(model: nn.Module, trainable: bool):
    for module in model.modules():
        if isinstance(module, MOTNFFNLayer):
            for proj in (module.gate_proj, module.up_proj, module.down_proj):
                gate = getattr(getattr(proj, "core", None), "gate", None)
                if gate is None:
                    continue
                for p in gate.parameters():
                    p.requires_grad_(trainable)


def _n_slide_for_proj(proj, cfg) -> int:
    layout = getattr(proj, "fitmotn_block_layout", None) or getattr(proj, "_fitmotn_block_layout", None)
    if isinstance(layout, dict) and layout.get("n_slide"):
        return max(1, min(int(proj.core.num_blocks), int(layout["n_slide"])))
    num_blocks = int(proj.core.num_blocks)
    ratio = float(getattr(cfg, "warmup_ratio", 0.5))
    n = int(round(num_blocks * ratio))
    return max(1, min(num_blocks, n))


def resolve_operator_block_layout(proj, cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    num_blocks = int(proj.core.num_blocks)
    positions = getattr(proj.core, "positions", None) or []
    pos_strategy = str((cfg or {}).get("pos_strategy", "random"))
    warmup_ratio = float((cfg or {}).get("warmup_ratio", 0.5))
    if pos_strategy == "warmup":
        n_slide = max(1, min(num_blocks, int(round(num_blocks * warmup_ratio))))
    else:
        n_slide = max(1, min(num_blocks, int(round(num_blocks * warmup_ratio))))
    slide_indices = list(range(n_slide))
    random_indices = list(range(n_slide, num_blocks))
    return {
        "n_total": num_blocks,
        "n_slide": len(slide_indices),
        "n_random": len(random_indices),
        "slide_indices": slide_indices,
        "random_indices": random_indices,
        "positions": positions,
        "pos_strategy": pos_strategy,
    }


def _compute_random_expert_scale(step: int, approx_cfg) -> float:
    if not bool(getattr(approx_cfg, "expert_warmup_scaling_enabled", False)):
        return 1.0
    total_steps = max(0, int(getattr(approx_cfg, "random_expert_warmup_steps", 0)))
    init_scale = float(getattr(approx_cfg, "random_expert_init_scale", 0.1))
    if total_steps <= 0 or step >= total_steps:
        return 1.0
    progress = max(0.0, min(1.0, float(step) / float(total_steps)))
    schedule = str(getattr(approx_cfg, "random_expert_warmup_schedule", "linear")).lower()
    if schedule == "cosine":
        eased = 0.5 - 0.5 * math.cos(math.pi * progress)
    else:
        eased = progress
    return float(init_scale + (1.0 - init_scale) * eased)


def configure_motn_expert_warmup_scaling(model: nn.Module, approx_cfg, global_step: int = 0) -> None:
    random_scale = _compute_random_expert_scale(int(global_step), approx_cfg)
    for _, module in iter_patched_motn_layers(model):
        state = {
            "enabled": bool(getattr(approx_cfg, "expert_warmup_scaling_enabled", False)),
            "global_step": int(global_step),
            "random_expert_init_scale": float(getattr(approx_cfg, "random_expert_init_scale", 0.1)),
            "random_expert_warmup_steps": int(getattr(approx_cfg, "random_expert_warmup_steps", 0)),
            "random_expert_warmup_schedule": str(getattr(approx_cfg, "random_expert_warmup_schedule", "linear")),
            "current_random_scale": float(random_scale),
        }
        for name in PROJ_NAMES:
            layout = getattr(module, "fitmotn_block_layout", {}).get(name) or resolve_operator_block_layout(getattr(module, name))
            state[name] = {
                "slide_indices": list(layout.get("slide_indices", [])),
                "random_indices": list(layout.get("random_indices", [])),
                "random_scale": float(random_scale),
                "n_slide": int(layout.get("n_slide", 0)),
                "n_random": int(layout.get("n_random", 0)),
            }
        module.fitmotn_expert_warmup_state = state


def update_motn_expert_warmup_scaling(model: nn.Module, approx_cfg, global_step: int) -> Dict[int, Dict[str, Any]]:
    summary: Dict[int, Dict[str, Any]] = {}
    random_scale = _compute_random_expert_scale(int(global_step), approx_cfg)
    for layer_idx, module in iter_patched_motn_layers(model):
        module.fitmotn_expert_warmup_state["enabled"] = bool(getattr(approx_cfg, "expert_warmup_scaling_enabled", False))
        module.fitmotn_expert_warmup_state["global_step"] = int(global_step)
        module.fitmotn_expert_warmup_state["current_random_scale"] = float(random_scale)
        layer_summary: Dict[str, Any] = {
            "enabled": bool(getattr(approx_cfg, "expert_warmup_scaling_enabled", False)),
            "global_step": int(global_step),
            "current_random_scale": float(random_scale),
        }
        for name in PROJ_NAMES:
            if name in module.fitmotn_expert_warmup_state:
                module.fitmotn_expert_warmup_state[name]["random_scale"] = float(random_scale)
                layer_summary[name] = dict(module.fitmotn_expert_warmup_state[name])
        summary[int(layer_idx)] = layer_summary
    return summary


def reset_motn_expert_warmup_scaling(model: nn.Module) -> None:
    for _, module in iter_patched_motn_layers(model):
        module.fitmotn_expert_warmup_state = {"enabled": False}


def build_motn_model_config(cfg) -> Dict[str, Any]:
    pos_strategy = cfg.model.pos_strategy
    if getattr(cfg.approx_init, "enabled", False) and str(getattr(cfg.approx_init, "subset_mode", "")) == "warmup_sliding_only" and str(pos_strategy).lower() != "warmup":
        pos_strategy = "warmup"
    return {
        "E": cfg.model.E,
        "d": cfg.model.d,
        "k_in": cfg.model.k_in,
        "gate_type": cfg.model.gate_type,
        "topk": cfg.model.topk,
        "temperature": cfg.model.temperature,
        "use_ste": cfg.model.use_ste,
        "jitter_eps": cfg.model.jitter_eps,
        "capacity_factor": cfg.model.capacity_factor,
        "min_capacity": cfg.model.min_capacity,
        "drop_tokens": cfg.model.drop_tokens,
        "drop_policy": cfg.model.drop_policy,
        "aux_coeff": cfg.model.aux_coeff,
        "zloss_coeff": cfg.model.zloss_coeff,
        "aux_mode": "ds",
        "pos_strategy": pos_strategy,
        "init_gamma": cfg.model.init_gamma,
        "warmup_ratio": cfg.model.warmup_ratio,
        "warmup_stride": cfg.model.warmup_stride,
        "dtype": tc.float32,
    }
