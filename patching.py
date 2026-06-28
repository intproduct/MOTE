from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List

import torch as tc
import torch.nn as nn
from .baselines.adtn_fixed import ADTNBaselineFFNLayer
from .model import MOTNFFNLayer


PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")
PATCHED_FFN_TYPES = (MOTNFFNLayer, ADTNBaselineFFNLayer)


def resolve_layer_idxs(n_layers: int, mode: str) -> List[int]:
    mode = str(mode).strip().lower()
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
    if mode.startswith("range:"):
        match = re.fullmatch(r"range:\s*([+-]?\d+)\s*-\s*([+-]?\d+)\s*", mode)
        if match is None:
            raise ValueError(f"Invalid layers_to_patch={mode!r}: expected range:START-END with integer bounds")
        start = int(match.group(1))
        end = int(match.group(2))
        if start < 0:
            raise ValueError(f"Invalid layers_to_patch={mode!r}: range START must be >= 0")
        if end < start:
            raise ValueError(f"Invalid layers_to_patch={mode!r}: range END must be >= START")
        if end >= n_layers:
            raise ValueError(f"Invalid layers_to_patch={mode!r}: range END must be < n_layers ({n_layers})")
        return list(range(start, end + 1))
    if mode.startswith("layers:"):
        raw_items = mode[len("layers:"):].split(",")
        items = [item.strip() for item in raw_items]
        if not items or any(item == "" for item in items):
            raise ValueError(f"Invalid layers_to_patch={mode!r}: layers list must be non-empty")
        layer_idxs: List[int] = []
        seen = set()
        for item in items:
            try:
                idx = int(item)
            except ValueError as exc:
                raise ValueError(f"Invalid layers_to_patch={mode!r}: layer index {item!r} is not an integer") from exc
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Invalid layers_to_patch={mode!r}: layer index {idx} must satisfy 0 <= idx < {n_layers}")
            if idx in seen:
                raise ValueError(f"Invalid layers_to_patch={mode!r}: duplicate layer index {idx}")
            seen.add(idx)
            layer_idxs.append(idx)
        return layer_idxs
    raise ValueError(f"Unsupported layers_to_patch={mode}")


def resolve_transformer_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise TypeError("model does not have transformer layers at model.model.layers or model.layers")


def patch_qwen_ffn_layers(model: nn.Module, layer_idxs: Iterable[int], motn_cfg: Dict[str, Any], device: tc.device, dtype=tc.float32, log=None) -> nn.Module:
    patch_cfg = dict(motn_cfg or {})
    patch_backend = str(patch_cfg.get("patch_backend", "motn")).lower()
    layers = resolve_transformer_layers(model)
    for idx in sorted(set(int(i) for i in layer_idxs)):
        layer = layers[idx]
        old_mlp = layer.mlp
        if patch_backend == "motn":
            layer.mlp = MOTNFFNLayer(old_mlp, patch_cfg, device=device, dtype=dtype, log=log, layer_idx=idx).to(device)
        elif patch_backend == "adtn_fixed":
            layer.mlp = ADTNBaselineFFNLayer(old_mlp, patch_cfg, device=device, dtype=dtype, log=log, layer_idx=idx).to(device)
        else:
            raise ValueError(f"Unsupported patch_backend={patch_backend!r}")
        layer.mlp.layer_idx = idx
        layer.mlp.patch_backend = patch_backend
        layer.mlp.fitmotn_block_layout = {}
        for name in PROJ_NAMES:
            layout = resolve_operator_block_layout(getattr(layer.mlp, name), patch_cfg)
            layer.mlp.fitmotn_block_layout[name] = layout
            setattr(getattr(layer.mlp, name), "fitmotn_block_layout", layout)
        layer.mlp.fitmotn_expert_warmup_state = {"enabled": False}
        if log is not None:
            log.info(f"[Patch] layer={idx} backend={patch_backend} replace mlp: {type(old_mlp)} -> {type(layer.mlp)}")
    return model


def set_trainable_patch_only(model: nn.Module, log=None) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    patched = 0
    for module in model.modules():
        if isinstance(module, PATCHED_FFN_TYPES):
            patched += 1
            for p in module.parameters():
                p.requires_grad_(True)
    if log is not None:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info(f"[Freeze] patched_layers={patched} | trainable={trainable} / total={total}")


def set_trainable_motn_only(model: nn.Module, log=None) -> None:
    set_trainable_patch_only(model, log=log)


def iter_patched_layers(model: nn.Module):
    items = []
    for module in model.modules():
        if isinstance(module, PATCHED_FFN_TYPES):
            items.append((int(getattr(module, "layer_idx", -1)), module))
    items.sort(key=lambda x: x[0])
    return items


def iter_patched_motn_layers(model: nn.Module):
    return iter_patched_layers(model)


def summarize_motn_gate_routers(model: nn.Module, patch_backend: str | None = None) -> Dict[str, Any]:
    if str(patch_backend or "motn").lower() != "motn":
        return {
            "gate_arch": None,
            "gate_hidden_dim": None,
            "resolved_gate_hidden_dim": None,
            "gate_activation": None,
            "gate_norm": None,
            "gate_dropout": None,
            "gate_mlp_bias": None,
            "gate_output_init_std": None,
            "gate_residual_delta_scale": None,
            "router_param_count": 0,
            "router_param_ratio_vs_blocks": None,
            "gate_router_summary": None,
        }

    routers = []
    block_params = 0
    resolved_by_proj: Dict[str, set] = {name: set() for name in PROJ_NAMES}
    for _, module in iter_patched_layers(model):
        if not isinstance(module, MOTNFFNLayer):
            continue
        for name in PROJ_NAMES:
            proj = getattr(module, name, None)
            core = getattr(proj, "core", None)
            gate = getattr(core, "gate", None)
            router = getattr(gate, "router", None)
            if router is not None:
                routers.append(router)
                hidden = getattr(router, "resolved_hidden_dim", None)
                if hidden is not None:
                    resolved_by_proj[name].add(int(hidden))
            blocks = getattr(core, "blocks", None)
            if blocks is not None:
                block_params += sum(p.numel() for p in blocks.parameters())

    if not routers:
        return {
            "gate_arch": None,
            "gate_hidden_dim": None,
            "resolved_gate_hidden_dim": None,
            "gate_activation": None,
            "gate_norm": None,
            "gate_dropout": None,
            "gate_mlp_bias": None,
            "gate_output_init_std": None,
            "gate_residual_delta_scale": None,
            "router_param_count": 0,
            "router_param_ratio_vs_blocks": None,
            "gate_router_summary": None,
        }

    first = routers[0]
    router_params = int(sum(sum(p.numel() for p in router.parameters()) for router in routers))
    resolved = {}
    for name, values in resolved_by_proj.items():
        if len(values) == 1:
            resolved[name] = next(iter(values))
        elif len(values) > 1:
            resolved[name] = sorted(values)
    if not resolved:
        resolved_hidden = None
    elif len(set(tuple(v) if isinstance(v, list) else v for v in resolved.values())) == 1:
        resolved_hidden = next(iter(resolved.values()))
    else:
        resolved_hidden = resolved
    summary = {
        "gate_arch": getattr(first, "gate_arch", None),
        "gate_hidden_dim": getattr(first, "gate_hidden_dim", None),
        "resolved_gate_hidden_dim": resolved_hidden,
        "gate_activation": getattr(first, "gate_activation", None),
        "gate_norm": getattr(first, "gate_norm", None),
        "gate_dropout": getattr(first, "gate_dropout", None),
        "gate_mlp_bias": getattr(first, "gate_mlp_bias", None),
        "gate_output_init_std": getattr(first, "gate_output_init_std", None),
        "gate_residual_delta_scale": getattr(first, "gate_residual_delta_scale", None),
        "router_param_count": router_params,
        "router_param_ratio_vs_blocks": None if block_params <= 0 else float(router_params) / float(block_params),
    }
    summary["gate_router_summary"] = dict(summary)
    return summary


def iter_trainable_params(model: nn.Module):
    for p in model.parameters():
        if p.requires_grad:
            yield p


def set_motn_temperature(model: nn.Module, temperature: float):
    for module in model.modules():
        if isinstance(module, PATCHED_FFN_TYPES):
            for proj in (module.gate_proj, module.up_proj, module.down_proj):
                try:
                    proj.core.gate.temperature = float(temperature)
                except Exception:
                    pass


def set_motn_gate_trainable(model: nn.Module, trainable: bool):
    for module in model.modules():
        if isinstance(module, PATCHED_FFN_TYPES):
            for proj in (module.gate_proj, module.up_proj, module.down_proj):
                gate = getattr(getattr(proj, "core", None), "gate", None)
                if gate is None:
                    continue
                for p in gate.parameters():
                    p.requires_grad_(trainable)


def set_motn_usage_tracking(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, PATCHED_FFN_TYPES):
            for proj in (module.gate_proj, module.up_proj, module.down_proj):
                core = getattr(proj, "core", None)
                if core is not None and hasattr(core, "set_usage_tracking_enabled"):
                    core.set_usage_tracking_enabled(enabled)


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
    for _, module in iter_patched_layers(model):
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
    for layer_idx, module in iter_patched_layers(model):
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
    for _, module in iter_patched_layers(model):
        module.fitmotn_expert_warmup_state = {"enabled": False}


def build_patch_model_config(cfg) -> Dict[str, Any]:
    pos_strategy = cfg.model.pos_strategy
    if getattr(cfg.approx_init, "enabled", False) and str(getattr(cfg.approx_init, "subset_mode", "")) == "warmup_sliding_only" and str(pos_strategy).lower() != "warmup":
        pos_strategy = "warmup"
    patch_backend = str(getattr(cfg.model, "patch_backend", "motn") or "motn").lower()
    patch_cfg = {
        "patch_backend": patch_backend,
        "E": cfg.model.E,
        "d": cfg.model.d,
        "k_in": cfg.model.k_in,
        "pos_strategy": pos_strategy,
        "init_gamma": cfg.model.init_gamma,
        "block_init_mode": getattr(cfg.model, "block_init_mode", "gamma_normal"),
        "block_init_std_scale": getattr(cfg.model, "block_init_std_scale", 1.0),
        "block_init_trunc_std": getattr(cfg.model, "block_init_trunc_std", 2.0),
        "warmup_ratio": cfg.model.warmup_ratio,
        "warmup_stride": cfg.model.warmup_stride,
        "dtype": tc.float32,
    }
    if patch_backend == "adtn_fixed":
        return patch_cfg
    patch_cfg.update({
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
        "gate_arch": cfg.model.gate_arch,
        "gate_hidden_dim": cfg.model.gate_hidden_dim,
        "gate_hidden_mult": cfg.model.gate_hidden_mult,
        "gate_hidden_min": cfg.model.gate_hidden_min,
        "gate_hidden_max": cfg.model.gate_hidden_max,
        "gate_activation": cfg.model.gate_activation,
        "gate_norm": cfg.model.gate_norm,
        "gate_dropout": cfg.model.gate_dropout,
        "gate_mlp_bias": cfg.model.gate_mlp_bias,
        "gate_output_init_std": cfg.model.gate_output_init_std,
        "gate_residual_delta_scale": cfg.model.gate_residual_delta_scale,
        "global_expert_enabled": cfg.model.global_expert_enabled,
        "global_expert_weight": cfg.model.global_expert_weight,
        "global_expert_init_scale": cfg.model.global_expert_init_scale,
        "global_expert_pos_strategy": cfg.model.global_expert_pos_strategy,
    })
    return patch_cfg


def build_motn_model_config(cfg) -> Dict[str, Any]:
    return build_patch_model_config(cfg)
