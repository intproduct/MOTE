from __future__ import annotations

import math
from typing import Any, Dict

import torch as tc
import torch.nn as nn

from .ADTN import MoTNLayer, block_init_stats_are_usable, dense_weight_init_stats
from .gate import gate_config_from_mapping
from .mixed_mixt import MixedMiXTLinear, QUADRANTS, projection_bonds
from .sparse_mixt import SparseMiXTLinear, resolve_main_dim


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
    gate_cfg = gate_config_from_mapping(cfg, data_dim=in_dim)
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


def _projection_ranks(sparse_cfg: Dict[str, Any], proj_name: str) -> Dict[str, int]:
    default_ranks = dict(sparse_cfg.get("ranks") or {"01": 8, "10": 8, "11": 8})
    projection = dict(sparse_cfg.get(f"{proj_name}_ranks") or default_ranks)
    missing = [key for key in ("01", "10", "11") if key not in projection]
    if missing:
        raise ValueError(f"sparse_mixt.{proj_name}_ranks is missing keys: {missing}")
    return {key: int(projection[key]) for key in ("01", "10", "11")}


class SparseMiXTFFNLayer(MOTNFFNLayer):
    """Qwen FFN using real-dimension SparseMiXTLinear projections."""

    def __init__(self, qwen_mlp: nn.Module, cfg: Dict[str, Any], device: tc.device, dtype=tc.float32, log=None, layer_idx: int = -1):
        nn.Module.__init__(self)
        if not (hasattr(qwen_mlp, "gate_proj") and hasattr(qwen_mlp, "up_proj") and hasattr(qwen_mlp, "down_proj")):
            raise TypeError(f"qwen_mlp does not look like QwenMLP, got: {type(qwen_mlp)}")
        self.hidden_size = int(qwen_mlp.gate_proj.in_features)
        self.intermediate_size = int(qwen_mlp.gate_proj.out_features)
        self.act = getattr(qwen_mlp, "act_fn", None) or nn.SiLU()
        self.logger = log
        self.layer_idx = int(layer_idx)
        self.fitmotn_block_layout = {}
        self.fitmotn_expert_warmup_state = {"enabled": False}

        sparse_cfg = dict(cfg.get("sparse_mixt") or {})
        d = int(cfg.get("d", 2))
        hidden_main = resolve_main_dim(self.hidden_size, int(sparse_cfg.get("hidden_main", 0)), d)
        intermediate_main = resolve_main_dim(
            self.intermediate_size,
            int(sparse_cfg.get("intermediate_main", 0)),
            d,
        )
        router_input_policy = str(sparse_cfg.get("router_input_policy", "main"))
        block_init_mode = str(cfg.get("block_init_mode", "gamma_normal") or "gamma_normal").strip().lower()
        init_stats = None
        if block_init_mode != "gamma_normal":
            init_stats = {
                "gate_proj": dense_weight_init_stats(qwen_mlp.gate_proj.weight),
                "up_proj": dense_weight_init_stats(qwen_mlp.up_proj.weight),
                "down_proj": dense_weight_init_stats(qwen_mlp.down_proj.weight),
            }
            for proj_name, stats in init_stats.items():
                _log_projection_block_init(log, layer_idx=layer_idx, proj_name=proj_name, backend="sparse_mixt", cfg=cfg, stats=stats)

        common = dict(cfg=cfg, router_input_policy=router_input_policy, dtype=dtype, device=device)
        self.gate_proj = SparseMiXTLinear(
            self.hidden_size,
            self.intermediate_size,
            ranks=_projection_ranks(sparse_cfg, "gate"),
            main_in_features=hidden_main,
            main_out_features=intermediate_main,
            init_stats=None if init_stats is None else init_stats["gate_proj"],
            **common,
        )
        self.up_proj = SparseMiXTLinear(
            self.hidden_size,
            self.intermediate_size,
            ranks=_projection_ranks(sparse_cfg, "up"),
            main_in_features=hidden_main,
            main_out_features=intermediate_main,
            init_stats=None if init_stats is None else init_stats["up_proj"],
            **common,
        )
        down_cfg = dict(cfg)
        down_cfg["k_in"] = int(self.gate_proj.core.k_out)
        self.down_proj = SparseMiXTLinear(
            self.intermediate_size,
            self.hidden_size,
            cfg=down_cfg,
            ranks=_projection_ranks(sparse_cfg, "down"),
            main_in_features=intermediate_main,
            main_out_features=hidden_main,
            router_input_policy=router_input_policy,
            dtype=dtype,
            device=device,
            init_stats=None if init_stats is None else init_stats["down_proj"],
        )

        boundary_init = str(sparse_cfg.get("boundary_init", "zero")).lower()
        if boundary_init in {"svd", "full_svd", "randomized_svd"}:
            method = "full_svd" if boundary_init == "svd" else boundary_init
            init_kwargs = {
                "boundary_method": method,
                "svd_oversampling": int(sparse_cfg.get("svd_oversampling", 8)),
                "svd_niter": int(sparse_cfg.get("svd_niter", 2)),
            }
            self.gate_proj.initialize_from_dense_weight(qwen_mlp.gate_proj.weight, **init_kwargs)
            self.up_proj.initialize_from_dense_weight(qwen_mlp.up_proj.weight, **init_kwargs)
            self.down_proj.initialize_from_dense_weight(qwen_mlp.down_proj.weight, **init_kwargs)
        elif boundary_init != "zero":
            raise ValueError(f"unsupported sparse_mixt.boundary_init={boundary_init!r}")

        if log is not None:
            log.info(
                f"[SparseMiXT] layer={layer_idx} hidden={self.hidden_size}={hidden_main}+{self.hidden_size-hidden_main} "
                f"intermediate={self.intermediate_size}={intermediate_main}+{self.intermediate_size-intermediate_main} "
                f"router={router_input_policy}"
            )

    def _forward_proj_with_scaling(self, proj: SparseMiXTLinear, x: tc.Tensor, proj_name: str) -> tc.Tensor:
        warmup_state = self.fitmotn_expert_warmup_state if isinstance(self.fitmotn_expert_warmup_state, dict) else {}
        proj_state = warmup_state.get(proj_name) if isinstance(warmup_state.get(proj_name), dict) else None
        enabled = bool(warmup_state.get("enabled")) and self.training and proj_state is not None
        if not enabled:
            return unwrap_y(proj(x))

        x2 = x.reshape(-1, proj.in_features)
        x0 = x2[:, : proj.main_in_features]
        gate_input = x0 if proj.router_input_policy == "main" else x2
        probs, mask, aux = proj.core.gate(gate_input.to(dtype=tc.float32))
        probs_scaled = self._apply_expert_scaling_to_routing(
            probs=probs,
            mask=mask,
            proj=proj,
            proj_name=proj_name,
        )
        return unwrap_y(proj(x, probs=probs_scaled, mask=mask, aux=aux))

    def forward_with_trace(self, x: tc.Tensor, trace=print) -> tc.Tensor:
        in_dtype = x.dtype
        trace(f"[SparseMiXTFFN] input={tuple(x.shape)} dtype={x.dtype}")
        g, _ = self.gate_proj.forward_with_trace(x, trace=trace)
        u, _ = self.up_proj.forward_with_trace(x, trace=trace)
        trace(f"[QwenMLP] h=act(gate) * up; activation={self.act.__class__.__name__} shape={tuple(g.shape)}")
        h = self.act(g) * u
        y, _ = self.down_proj.forward_with_trace(h, trace=trace)
        trace(f"[SparseMiXTFFN] output={tuple(y.shape)} restore_dtype={in_dtype}")
        return y.to(dtype=in_dtype)


def _mixed_quadrant_init_stats(weight: tc.Tensor, main_in: int, main_out: int) -> Dict[str, Dict[str, object]]:
    return {
        "m00": dense_weight_init_stats(weight[:main_out, :main_in]),
        "m01": dense_weight_init_stats(weight[:main_out, main_in:]),
        "m10": dense_weight_init_stats(weight[main_out:, :main_in]),
        "m11": dense_weight_init_stats(weight[main_out:, main_in:]),
    }


class MixedMiXTFFNLayer(MOTNFFNLayer):
    """Qwen FFN whose four real-dimension quadrants are all MiXT operators."""

    def __init__(self, qwen_mlp: nn.Module, cfg: Dict[str, Any], device: tc.device, dtype=tc.float32, log=None, layer_idx: int = -1):
        nn.Module.__init__(self)
        if not (hasattr(qwen_mlp, "gate_proj") and hasattr(qwen_mlp, "up_proj") and hasattr(qwen_mlp, "down_proj")):
            raise TypeError(f"qwen_mlp does not look like QwenMLP, got: {type(qwen_mlp)}")
        self.hidden_size = int(qwen_mlp.gate_proj.in_features)
        self.intermediate_size = int(qwen_mlp.gate_proj.out_features)
        self.act = getattr(qwen_mlp, "act_fn", None) or nn.SiLU()
        self.logger = log
        self.layer_idx = int(layer_idx)
        self.fitmotn_block_layout = {}
        self.fitmotn_expert_warmup_state = {"enabled": False}

        mixed_cfg = dict(cfg.get("mixed_mixt") or {})
        d = int(cfg.get("d", 2))
        hidden_main = resolve_main_dim(self.hidden_size, int(mixed_cfg.get("hidden_main", 0)), d)
        intermediate_main = resolve_main_dim(
            self.intermediate_size,
            int(mixed_cfg.get("intermediate_main", 0)),
            d,
        )
        router_input_policy = str(mixed_cfg.get("router_input_policy", "full_real"))
        block_init_mode = str(cfg.get("block_init_mode", "gamma_normal") or "gamma_normal").strip().lower()
        projection_stats = None
        if block_init_mode != "gamma_normal":
            projection_stats = {
                "gate_proj": _mixed_quadrant_init_stats(qwen_mlp.gate_proj.weight, hidden_main, intermediate_main),
                "up_proj": _mixed_quadrant_init_stats(qwen_mlp.up_proj.weight, hidden_main, intermediate_main),
                "down_proj": _mixed_quadrant_init_stats(qwen_mlp.down_proj.weight, intermediate_main, hidden_main),
            }
            for proj_name, weight in (
                ("gate_proj", qwen_mlp.gate_proj.weight),
                ("up_proj", qwen_mlp.up_proj.weight),
                ("down_proj", qwen_mlp.down_proj.weight),
            ):
                _log_projection_block_init(
                    log,
                    layer_idx=layer_idx,
                    proj_name=proj_name,
                    backend="mixed_mixt",
                    cfg=cfg,
                    stats=dense_weight_init_stats(weight),
                )

        def quadrant_stats(proj_name: str):
            return None if projection_stats is None else projection_stats[proj_name]

        common = dict(cfg=cfg, router_input_policy=router_input_policy, dtype=dtype, device=device)
        self.gate_proj = MixedMiXTLinear(
            self.hidden_size,
            self.intermediate_size,
            bonds=projection_bonds(mixed_cfg, "gate"),
            main_in_features=hidden_main,
            main_out_features=intermediate_main,
            init_stats=quadrant_stats("gate_proj"),
            **common,
        )
        self.up_proj = MixedMiXTLinear(
            self.hidden_size,
            self.intermediate_size,
            bonds=projection_bonds(mixed_cfg, "up"),
            main_in_features=hidden_main,
            main_out_features=intermediate_main,
            init_stats=quadrant_stats("up_proj"),
            **common,
        )
        self.down_proj = MixedMiXTLinear(
            self.intermediate_size,
            self.hidden_size,
            bonds=projection_bonds(mixed_cfg, "down"),
            main_in_features=intermediate_main,
            main_out_features=hidden_main,
            init_stats=quadrant_stats("down_proj"),
            **common,
        )

        dense_init = str(mixed_cfg.get("dense_init", "none")).strip().lower()
        if dense_init == "full_site_exact":
            self.gate_proj.initialize_from_dense_weight_exact(qwen_mlp.gate_proj.weight)
            self.up_proj.initialize_from_dense_weight_exact(qwen_mlp.up_proj.weight)
            self.down_proj.initialize_from_dense_weight_exact(qwen_mlp.down_proj.weight)
        elif dense_init != "none":
            raise ValueError(f"unsupported mixed_mixt.dense_init={dense_init!r}")

        if log is not None:
            log.info(
                f"[MixedMiXT] layer={layer_idx} hidden={self.hidden_size}={hidden_main}+{self.hidden_size-hidden_main} "
                f"intermediate={self.intermediate_size}={intermediate_main}+{self.intermediate_size-intermediate_main} "
                f"router={router_input_policy} shared_gate=True"
            )

    def _forward_proj_with_scaling(self, proj: MixedMiXTLinear, x: tc.Tensor, proj_name: str) -> tc.Tensor:
        warmup_state = self.fitmotn_expert_warmup_state if isinstance(self.fitmotn_expert_warmup_state, dict) else {}
        proj_state = warmup_state.get(proj_name) if isinstance(warmup_state.get(proj_name), dict) else None
        enabled = bool(warmup_state.get("enabled")) and self.training and proj_state is not None
        if not enabled:
            return unwrap_y(proj(x))

        x2 = x.reshape(-1, proj.in_features)
        x0 = x2[:, : proj.main_in_features]
        gate_input = x2 if proj.router_input_policy == "full_real" else x0
        probs, mask, aux = proj.core.gate(gate_input.to(dtype=tc.float32))
        probs_scaled = self._apply_expert_scaling_to_routing(
            probs=probs,
            mask=mask,
            proj=proj,
            proj_name=proj_name,
        )
        return unwrap_y(proj(x, probs=probs_scaled, mask=mask, aux=aux))

    def forward_with_trace(self, x: tc.Tensor, trace=print) -> tc.Tensor:
        in_dtype = x.dtype
        trace(f"[MixedMiXTFFN] input={tuple(x.shape)} dtype={x.dtype}")
        g, _ = self.gate_proj.forward_with_trace(x, trace=trace)
        u, _ = self.up_proj.forward_with_trace(x, trace=trace)
        trace(f"[QwenMLP] h=act(gate) * up; activation={self.act.__class__.__name__} shape={tuple(g.shape)}")
        h = self.act(g) * u
        y, _ = self.down_proj.forward_with_trace(h, trace=trace)
        trace(f"[MixedMiXTFFN] output={tuple(y.shape)} restore_dtype={in_dtype}")
        return y.to(dtype=in_dtype)
