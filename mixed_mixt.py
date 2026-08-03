from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, Mapping, Optional

import torch as tc
import torch.nn as nn

from .ADTN import GatedADTNLayer, q_from_dim
from .gate import gate_config_from_mapping, gate_factory_config
from .sparse_mixt import resolve_main_dim


TraceFn = Callable[[str], None]
QUADRANTS = ("m00", "m01", "m10", "m11")


def _resolve_bond(q_in: int, q_out: int, requested: int, fallback: int) -> int:
    """Resolve k_in while guaranteeing 0 <= k_out <= q_out."""
    lower = max(1, int(q_in) - int(q_out))
    candidate = int(requested) if int(requested) > 0 else int(fallback)
    k_in = max(lower, min(int(q_in), candidate))
    k_out = k_in + int(q_out) - int(q_in)
    if not (1 <= k_in <= int(q_in)) or not (0 <= k_out <= int(q_out)):
        raise ValueError(
            f"invalid MiXT bond q_in={q_in} q_out={q_out} "
            f"requested_k_in={requested} resolved=({k_in},{k_out})"
        )
    return k_in


def _projection_bonds(mixed_cfg: Mapping[str, Any], proj_name: str) -> Dict[str, int]:
    default = dict(mixed_cfg.get("bonds") or {})
    projection = dict(mixed_cfg.get(f"{proj_name}_bonds") or default)
    return {name: int(projection.get(name, 0)) for name in QUADRANTS}


class MixedMiXTCore(nn.Module):
    """Four MiXT quadrants driven by one shared routing decision."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        main_in_features: int,
        main_out_features: int,
        cfg: Mapping[str, Any],
        bonds: Mapping[str, int],
        router_input_policy: str,
        dtype: tc.dtype,
        device: Optional[tc.device],
        init_stats: Optional[Mapping[str, Dict[str, object]]] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.main_in_features = int(main_in_features)
        self.main_out_features = int(main_out_features)
        self.tail_in_features = self.in_features - self.main_in_features
        self.tail_out_features = self.out_features - self.main_out_features
        if self.tail_in_features <= 0 or self.tail_out_features <= 0:
            raise ValueError(
                "mixed_mixt requires non-empty main and tail dimensions; "
                f"got input={self.main_in_features}+{self.tail_in_features}, "
                f"output={self.main_out_features}+{self.tail_out_features}"
            )

        self.d = int(cfg.get("d", 2))
        self.router_input_policy = str(router_input_policy).strip().lower()
        if self.router_input_policy not in {"main", "full_real"}:
            raise ValueError(f"unsupported router_input_policy={router_input_policy!r}")
        self.requested_num_blocks = int(cfg["E"])

        specs = {
            "m00": (self.main_in_features, self.main_out_features),
            "m01": (self.tail_in_features, self.main_out_features),
            "m10": (self.main_in_features, self.tail_out_features),
            "m11": (self.tail_in_features, self.tail_out_features),
        }
        fallback = int(cfg.get("k_in", 3))
        gate_dim = self.in_features if self.router_input_policy == "full_real" else self.main_in_features
        base_gate_cfg = gate_config_from_mapping(cfg, data_dim=gate_dim)
        quadrants: Dict[str, GatedADTNLayer] = {}
        for offset, name in enumerate(QUADRANTS):
            dim_in, dim_out = specs[name]
            q_in = q_from_dim(dim_in, self.d)
            q_out = q_from_dim(dim_out, self.d)
            k_in = _resolve_bond(q_in, q_out, int(bonds.get(name, 0)), fallback)
            core = GatedADTNLayer(
                dim_input=dim_in,
                dim_output=dim_out,
                num_blocks=self.requested_num_blocks,
                d=self.d,
                k_in=k_in,
                gate_config=base_gate_cfg,
                entropy_coeff=float(cfg.get("entropy_coeff", 0.0)),
                pos_strategy=str(cfg.get("pos_strategy", "random")),
                dtype=cfg.get("dtype", dtype),
                device=device,
                init_gamma=float(cfg.get("init_gamma", 0.6)),
                seed=int(cfg.get("seed", 0)) + offset,
                warmup_ratio=float(cfg.get("warmup_ratio", 0.5)),
                warmup_stride=int(cfg.get("warmup_stride", 1)),
                global_expert_enabled=bool(cfg.get("global_expert_enabled", False)),
                global_expert_weight=float(cfg.get("global_expert_weight", 1.0)),
                global_expert_init_scale=float(cfg.get("global_expert_init_scale", 1.0)),
                global_expert_pos_strategy=str(cfg.get("global_expert_pos_strategy", "spread")),
                block_init_mode=str(cfg.get("block_init_mode", "gamma_normal")),
                block_init_std_scale=float(cfg.get("block_init_std_scale", 1.0)),
                block_init_trunc_std=float(cfg.get("block_init_trunc_std", 2.0)),
                init_stats=None if init_stats is None else init_stats.get(name),
            )
            if core.num_blocks != self.requested_num_blocks:
                raise ValueError(
                    f"mixed_mixt shared routing requires E={self.requested_num_blocks} in every quadrant, "
                    f"but {name} only constructed {core.num_blocks}; choose a bond/pos_strategy with "
                    "enough unique expert positions"
                )
            # Routing is owned by this MixedMiXTCore. Quadrants receive the
            # shared probs/mask explicitly and must not retain unused routers.
            core.gate = None
            quadrants[name] = core
        self.quadrants = nn.ModuleDict(quadrants)
        self.num_blocks = self.requested_num_blocks

        shared_gate_cfg = replace(base_gate_cfg, num_experts=self.num_blocks, data_dim=gate_dim)
        self.gate = gate_factory_config(shared_gate_cfg).to(dtype=tc.float32, device=device)
        self.last_aux_dict = None

        reference = self.quadrants["m00"]
        self.q_in = reference.q_in
        self.q_out = reference.q_out
        self.k_in = reference.k_in
        self.k_out = reference.k_out
        self.global_expert_enabled = bool(cfg.get("global_expert_enabled", False))

    @property
    def blocks(self):
        """Compatibility view used by shared expert-layout controls."""
        return self.quadrants["m00"].blocks

    @property
    def positions(self):
        return self.quadrants["m00"].positions

    @property
    def global_block(self):
        return self.quadrants["m00"].global_block

    @property
    def last_usage(self):
        return self.quadrants["m00"].last_usage

    def set_usage_tracking_enabled(self, enabled: bool) -> None:
        if hasattr(self.gate, "set_usage_tracking_enabled"):
            self.gate.set_usage_tracking_enabled(enabled)
        for quadrant in self.quadrants.values():
            quadrant.set_usage_tracking_enabled(enabled)

    def reset_runtime_usage_cache(self) -> None:
        self.last_aux_dict = None
        if hasattr(self.gate, "reset_runtime_usage_cache"):
            self.gate.reset_runtime_usage_cache()
        for quadrant in self.quadrants.values():
            quadrant.reset_runtime_usage_cache()

    def collect_runtime_usage_tensors(self):
        stats = dict(self.quadrants["m00"].collect_runtime_usage_tensors())
        gate_stats = self.gate.collect_runtime_usage_tensors() if hasattr(self.gate, "collect_runtime_usage_tensors") else {}
        for key, value in gate_stats.items():
            if value is not None:
                stats[key] = value
        return stats

    def materialize_usage_report(self):
        report = dict(self.quadrants["m00"].materialize_usage_report())
        if hasattr(self.gate, "materialize_usage_report"):
            gate_report = self.gate.materialize_usage_report()
            for key, value in gate_report.items():
                if value is not None:
                    report[key] = value
        report["shared_router"] = True
        report["quadrants"] = list(QUADRANTS)
        return report

    def _route(self, x: tc.Tensor, x0: tc.Tensor):
        gate_input = x if self.router_input_policy == "full_real" else x0
        return self.gate(gate_input.to(dtype=tc.float32))

    def forward(
        self,
        x: tc.Tensor,
        *,
        probs: Optional[tc.Tensor] = None,
        mask: Optional[tc.Tensor] = None,
        aux: Optional[Dict] = None,
        idx_w=None,
        dense: bool = False,
        use_global_expert: Optional[bool] = None,
    ):
        x2 = x.reshape(-1, self.in_features)
        x0 = x2[:, : self.main_in_features]
        x1 = x2[:, self.main_in_features :]
        if probs is None and mask is None and idx_w is None and not dense:
            probs, mask, aux = self._route(x2, x0)
        self.last_aux_dict = aux
        kwargs = {
            "probs": probs,
            "mask": mask,
            "aux": aux,
            "idx_w": idx_w,
            "dense": dense,
            "use_global_expert": use_global_expert,
        }
        z00 = self.quadrants["m00"](x0, **kwargs)
        z01 = self.quadrants["m01"](x1, **kwargs)
        z10 = self.quadrants["m10"](x0, **kwargs)
        z11 = self.quadrants["m11"](x1, **kwargs)
        y0 = z00 + z01
        y1 = z10 + z11
        return tc.cat((y0, y1), dim=-1), aux


class MixedMiXTLinear(nn.Module):
    """Real-dimension four-quadrant MiXT operator with one shared gate."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        cfg: Mapping[str, Any],
        bonds: Mapping[str, int],
        main_in_features: int = 0,
        main_out_features: int = 0,
        router_input_policy: str = "full_real",
        dtype: tc.dtype = tc.float32,
        device: Optional[tc.device] = None,
        init_stats: Optional[Mapping[str, Dict[str, object]]] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.d = int(cfg.get("d", 2))
        mixed_cfg = dict(cfg.get("mixed_mixt") or {})
        backend_version = int(mixed_cfg.get("backend_version", 1))
        if backend_version != 1:
            raise ValueError(f"unsupported mixed_mixt backend_version={backend_version}; expected 1")
        self.main_in_features = resolve_main_dim(self.in_features, main_in_features, self.d)
        self.main_out_features = resolve_main_dim(self.out_features, main_out_features, self.d)
        self.tail_in_features = self.in_features - self.main_in_features
        self.tail_out_features = self.out_features - self.main_out_features
        self.router_input_policy = str(router_input_policy).strip().lower()
        self.core = MixedMiXTCore(
            in_features=self.in_features,
            out_features=self.out_features,
            main_in_features=self.main_in_features,
            main_out_features=self.main_out_features,
            cfg=cfg,
            bonds=bonds,
            router_input_policy=self.router_input_policy,
            dtype=dtype,
            device=device,
            init_stats=init_stats,
        )
        self.in_pad = self.in_features
        self.out_pad = self.out_features
        self.q_in = self.core.q_in
        self.q_out = self.core.q_out

    def forward(self, x: tc.Tensor, **kwargs):
        if int(x.shape[-1]) != self.in_features:
            raise ValueError(f"expected input last dim {self.in_features}, got {int(x.shape[-1])}")
        orig_shape = x.shape
        y2, aux = self.core(x.reshape(-1, self.in_features), **kwargs)
        return y2.reshape(*orig_shape[:-1], self.out_features), aux

    def forward_with_trace(self, x: tc.Tensor, trace: TraceFn = print, **kwargs):
        trace(
            f"[MixedMiXTLinear] real={self.in_features}->{self.out_features} "
            f"split={self.main_in_features}+{self.tail_in_features} -> "
            f"{self.main_out_features}+{self.tail_out_features}"
        )
        trace(
            f"[shared routing] gate={self.core.gate.__class__.__name__} "
            f"topk={getattr(self.core.gate, 'k', 'soft')} experts={self.core.num_blocks} "
            f"router_input={self.router_input_policy} calls=1"
        )
        for name in QUADRANTS:
            quadrant = self.core.quadrants[name]
            trace(
                f"[{name}] {quadrant.dim_input}->{quadrant.dim_output} "
                f"q={quadrant.q_in}->{quadrant.q_out} "
                f"bond={quadrant.k_in}->{quadrant.k_out}"
            )
        trace("[compute] y0=M00(x0)+M01(x1); y1=M10(x0)+M11(x1)")
        y, aux = self.forward(x, **kwargs)
        trace(f"[concat] output={tuple(y.shape)} padding=False crop=False")
        return y, aux

    @tc.no_grad()
    def initialize_from_dense_weight_exact(self, weight: tc.Tensor) -> Dict[str, Dict[str, object]]:
        """Copy four dense quadrants into full-site single-expert MiXT blocks."""
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(f"unexpected dense weight shape: {tuple(weight.shape)}")
        mi, mo = self.main_in_features, self.main_out_features
        pieces = {
            "m00": weight[:mo, :mi],
            "m01": weight[:mo, mi:],
            "m10": weight[mo:, :mi],
            "m11": weight[mo:, mi:],
        }
        report: Dict[str, Dict[str, object]] = {}
        for name, source in pieces.items():
            quadrant = self.core.quadrants[name]
            if quadrant.num_blocks != 1 or quadrant.k_in != quadrant.q_in or quadrant.k_out != quadrant.q_out:
                raise ValueError(
                    f"exact initialization of {name} requires E=1 and full-site bond "
                    f"({quadrant.q_in}->{quadrant.q_out}), got E={quadrant.num_blocks} "
                    f"bond={quadrant.k_in}->{quadrant.k_out}"
                )
            target = source.detach().to(device=quadrant.blocks[0].U.device, dtype=quadrant.blocks[0].U.dtype)
            quadrant.blocks[0].U.copy_(target.transpose(0, 1).reshape_as(quadrant.blocks[0].U))
            report[name] = {
                "exact": True,
                "shape": list(source.shape),
                "bond": [quadrant.k_in, quadrant.k_out],
            }
        return report


def projection_bonds(mixed_cfg: Mapping[str, Any], proj_name: str) -> Dict[str, int]:
    return _projection_bonds(mixed_cfg, proj_name)
