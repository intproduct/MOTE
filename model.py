from __future__ import annotations

import math
from typing import Any, Dict

import torch as tc
import torch.nn as nn

from .ADTN import MoTNLayer, block_init_stats_are_usable, dense_weight_init_stats
from .gate import GateConfig


def unwrap_y(out):
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _log_projection_block_init(log, *, layer_idx: int, proj_name: str, backend: str, cfg: Dict[str, Any], stats: Dict[str, Any]) -> None:
    if log is None:
        return
    mode = str(cfg.get("block_init_mode", "gamma_normal"))
    message = (
        f"[BlockInit] layer={layer_idx} proj={proj_name} backend={backend} "
        f"mode={mode} mean={float(stats.get('mean', float('nan'))):.6g} "
        f"std={float(stats.get('std', float('nan'))):.6g} "
        f"min={float(stats.get('min', float('nan'))):.6g} "
        f"max={float(stats.get('max', float('nan'))):.6g} "
        f"numel={int(stats.get('numel', 0))} "
        f"std_scale={float(cfg.get('block_init_std_scale', 1.0)):.6g} "
        f"trunc_std={float(cfg.get('block_init_trunc_std', 2.0)):.6g}"
    )
    if mode != "gamma_normal" and not block_init_stats_are_usable(stats):
        try:
            log.warning(message + " fallback=gamma_normal reason=invalid_stats")
        except Exception:
            pass
        return
    try:
        log.info(message)
    except Exception:
        pass


def build_motn_layer(*, in_dim: int, out_dim: int, cfg: Dict[str, Any], device: tc.device, dtype=tc.float32, log=None, init_stats=None) -> MoTNLayer:
    gate_cfg = GateConfig(
        gate_type=str(cfg["gate_type"]),
        data_dim=int(in_dim),
        num_experts=int(cfg["E"]),
        k=int(cfg["topk"]),
        temperature=float(cfg.get("temperature", 1.0)),
        use_ste=bool(cfg.get("use_ste", True)),
        jitter_eps=float(cfg.get("jitter_eps", 0.0)),
        capacity_factor=float(cfg.get("capacity_factor", 1.0)),
        min_capacity=int(cfg.get("min_capacity", 8)),
        drop_tokens=bool(cfg.get("drop_tokens", True)),
        drop_policy=str(cfg.get("drop_policy", "probs")),
        aux_coeff=float(cfg.get("aux_coeff", 1e-2)),
        zloss_coeff=float(cfg.get("zloss_coeff", 0.0)),
        aux_mode=str(cfg.get("aux_mode", "ds")),
        gate_arch=str(cfg.get("gate_arch", "linear")),
        gate_hidden_dim=int(cfg.get("gate_hidden_dim", 0)),
        gate_hidden_mult=float(cfg.get("gate_hidden_mult", 0.0625)),
        gate_hidden_min=int(cfg.get("gate_hidden_min", 64)),
        gate_hidden_max=int(cfg.get("gate_hidden_max", 256)),
        gate_activation=str(cfg.get("gate_activation", "silu")),
        gate_norm=str(cfg.get("gate_norm", "none")),
        gate_dropout=float(cfg.get("gate_dropout", 0.0)),
        gate_mlp_bias=bool(cfg.get("gate_mlp_bias", True)),
        gate_output_init_std=float(cfg.get("gate_output_init_std", 1e-3)),
        gate_residual_delta_scale=float(cfg.get("gate_residual_delta_scale", 1.0)),
    )
    motn = MoTNLayer(
        in_dim,
        out_dim,
        d=int(cfg.get("d", 2)),
        num_blocks=int(cfg["E"]),
        k_in=int(cfg.get("k_in", 3)),
        gate_config=gate_cfg,
        entropy_coeff=float(cfg.get("entropy_coeff", 0.0)),
        pos_strategy=str(cfg.get("pos_strategy", "random")),
        dtype=cfg.get("dtype", dtype),
        device=device,
        init_gamma=float(cfg.get("init_gamma", 0.6)),
        seed=int(cfg.get("seed", 0)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.5)),
        warmup_stride=int(cfg.get("warmup_stride", 1)),
        global_expert_enabled=bool(cfg.get("global_expert_enabled", False)),
        global_expert_weight=float(cfg.get("global_expert_weight", 1.0)),
        global_expert_init_scale=float(cfg.get("global_expert_init_scale", 1.0)),
        global_expert_pos_strategy=str(cfg.get("global_expert_pos_strategy", "spread")),
        block_init_mode=str(cfg.get("block_init_mode", "gamma_normal")),
        block_init_std_scale=float(cfg.get("block_init_std_scale", 1.0)),
        block_init_trunc_std=float(cfg.get("block_init_trunc_std", 2.0)),
        init_stats=init_stats,
    ).to(device)
    if log is not None:
        try:
            log.info(
                f"[MoTN] build in={in_dim} out={out_dim} E={motn.core.num_blocks} "
                f"k_in={motn.core.k_in} k_out={motn.core.k_out} gate={gate_cfg.gate_type}"
            )
            gate = getattr(motn.core, "gate", None)
            router = getattr(gate, "router", None)
            if router is not None:
                hidden = getattr(router, "resolved_hidden_dim", None)
                hidden_text = "-" if hidden is None else str(hidden)
                log.info(
                    f"[Gate] arch={router.gate_arch} hidden={hidden_text} "
                    f"activation={router.gate_activation} norm={router.gate_norm} "
                    f"params={router.router_param_count}"
                )
        except Exception:
            pass
    return motn


class MOTNFFNLayer(nn.Module):
    def __init__(self, qwen_mlp: nn.Module, cfg: Dict[str, Any], device: tc.device, dtype=tc.float32, log=None, layer_idx: int = -1):
        super().__init__()
        if not (hasattr(qwen_mlp, "gate_proj") and hasattr(qwen_mlp, "up_proj") and hasattr(qwen_mlp, "down_proj")):
            raise TypeError(f"qwen_mlp does not look like QwenMLP, got: {type(qwen_mlp)}")
        self.hidden_size = int(qwen_mlp.gate_proj.in_features)
        self.intermediate_size = int(qwen_mlp.gate_proj.out_features)
        self.act = getattr(qwen_mlp, "act_fn", None) or nn.SiLU()
        self.logger = log
        block_init_mode = str(cfg.get("block_init_mode", "gamma_normal") or "gamma_normal").strip().lower()
        init_stats = None
        if block_init_mode != "gamma_normal":
            init_stats = {
                "gate_proj": dense_weight_init_stats(qwen_mlp.gate_proj.weight),
                "up_proj": dense_weight_init_stats(qwen_mlp.up_proj.weight),
                "down_proj": dense_weight_init_stats(qwen_mlp.down_proj.weight),
            }
            for proj_name, stats in init_stats.items():
                _log_projection_block_init(log, layer_idx=layer_idx, proj_name=proj_name, backend="motn", cfg=cfg, stats=stats)
        self.gate_proj = build_motn_layer(
            in_dim=self.hidden_size,
            out_dim=self.intermediate_size,
            cfg=cfg,
            device=device,
            dtype=dtype,
            log=log,
            init_stats=None if init_stats is None else init_stats["gate_proj"],
        )
        self.up_proj = build_motn_layer(
            in_dim=self.hidden_size,
            out_dim=self.intermediate_size,
            cfg=cfg,
            device=device,
            dtype=dtype,
            log=log,
            init_stats=None if init_stats is None else init_stats["up_proj"],
        )
        down_cfg = dict(cfg)
        down_cfg["k_in"] = int(self.gate_proj.core.k_out)
        self.down_proj = build_motn_layer(
            in_dim=self.intermediate_size,
            out_dim=self.hidden_size,
            cfg=down_cfg,
            device=device,
            dtype=dtype,
            log=log,
            init_stats=None if init_stats is None else init_stats["down_proj"],
        )
        self.layer_idx = int(layer_idx)
        self.fitmotn_block_layout = {}
        self.fitmotn_expert_warmup_state = {"enabled": False}

    def _forward_proj_with_scaling(self, proj: MoTNLayer, x: tc.Tensor, proj_name: str) -> tc.Tensor:
        warmup_state = self.fitmotn_expert_warmup_state if isinstance(self.fitmotn_expert_warmup_state, dict) else {}
        proj_state = warmup_state.get(proj_name) if isinstance(warmup_state.get(proj_name), dict) else None
        enabled = bool(warmup_state.get("enabled")) and self.training and proj_state is not None
        if not enabled:
            return unwrap_y(proj(x))

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]).to(dtype=tc.float32)
        if proj.in_pad != proj.in_features:
            x2 = nn.functional.pad(x2, (0, proj.in_pad - proj.in_features))

        probs, mask, aux = proj.core.gate(x2)
        probs_scaled = self._apply_expert_scaling_to_routing(
            probs=probs,
            mask=mask,
            proj=proj,
            proj_name=proj_name,
        )
        out = proj.core(x2, probs=probs_scaled, mask=mask, aux=aux)
        y2 = unwrap_y(out).reshape(x2.shape[0], -1)
        if proj.out_pad != proj.out_features:
            y2 = y2[:, :proj.out_features]
        return y2.reshape(*orig_shape[:-1], proj.out_features)

    def _apply_expert_scaling_to_routing(self, probs: tc.Tensor, mask: tc.Tensor | None, proj: MoTNLayer, proj_name: str) -> tc.Tensor:
        warmup_state = self.fitmotn_expert_warmup_state
        proj_state = warmup_state.get(proj_name) or {}
        random_scale = float(proj_state.get("random_scale", 1.0))
        if math.isclose(random_scale, 1.0):
            return probs

        layout = self.fitmotn_block_layout.get(proj_name) or {}
        scales = tc.ones(int(proj.core.num_blocks), device=probs.device, dtype=tc.float32)
        random_indices = list(layout.get("random_indices", []))
        if random_indices:
            ridx = tc.tensor(random_indices, device=probs.device, dtype=tc.long)
            scales[ridx] = random_scale
        probs_fp32 = probs.to(tc.float32)
        if mask is None:
            weighted = probs_fp32 * scales.view(*([1] * (probs_fp32.ndim - 1)), -1)
        else:
            mask_fp32 = mask.to(tc.float32)
            weighted = probs_fp32 * mask_fp32
            weighted = weighted * scales.view(*([1] * (weighted.ndim - 1)), -1)
        denom = weighted.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return (weighted / denom).to(dtype=probs.dtype)

    def forward(self, x: tc.Tensor) -> tc.Tensor:
        in_dtype = x.dtype
        g = self._forward_proj_with_scaling(self.gate_proj, x, "gate_proj")
        u = self._forward_proj_with_scaling(self.up_proj, x, "up_proj")
        h = self.act(g) * u
        y = self._forward_proj_with_scaling(self.down_proj, h, "down_proj")
        return y.to(dtype=in_dtype)
