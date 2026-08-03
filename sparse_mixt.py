from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, Mapping, Optional

import torch as tc
import torch.nn as nn

from .ADTN import GatedADTNLayer, is_pow_of
from .gate import gate_config_from_mapping, gate_factory_config


TraceFn = Callable[[str], None]


def largest_power_of_d_leq(dim: int, d: int = 2) -> int:
    """Return the largest exact d**q not greater than ``dim``."""
    dim = int(dim)
    d = int(d)
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if d <= 1:
        raise ValueError(f"d must be greater than one, got {d}")
    value = 1
    while value * d <= dim:
        value *= d
    return value


def resolve_main_dim(dim: int, requested: int, d: int) -> int:
    main = int(requested) if int(requested) > 0 else largest_power_of_d_leq(dim, d)
    if main > int(dim):
        raise ValueError(f"main dimension {main} exceeds real dimension {dim}")
    if not is_pow_of(main, d):
        raise ValueError(f"main dimension {main} must be an exact power of d={d}")
    return main


def _align_input_dtype(x: tc.Tensor, module: nn.Module) -> tc.Tensor:
    ref = next(module.parameters(), None)
    if ref is not None and x.is_floating_point() and x.dtype != ref.dtype:
        return x.to(dtype=ref.dtype)
    return x


class LowRankLinear(nn.Module):
    """Bias-free A(Bx) boundary map with an auditable two-GEMM layout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        dtype: tc.dtype = tc.float32,
        device: Optional[tc.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        if min(self.in_features, self.out_features, self.rank) <= 0:
            raise ValueError(
                "LowRankLinear requires positive in_features, out_features, and rank; "
                f"got {self.in_features}, {self.out_features}, {self.rank}"
            )
        self.B = nn.Linear(self.in_features, self.rank, bias=False, dtype=dtype, device=device)
        self.A = nn.Linear(self.rank, self.out_features, bias=False, dtype=dtype, device=device)
        nn.init.kaiming_uniform_(self.B.weight, a=5 ** 0.5)
        nn.init.zeros_(self.A.weight)

    def forward(self, x: tc.Tensor) -> tc.Tensor:
        x = _align_input_dtype(x, self.B)
        return self.A(self.B(x))

    @tc.no_grad()
    def set_from_dense_weight(
        self,
        weight: tc.Tensor,
        *,
        method: str = "auto",
        oversampling: int = 8,
        niter: int = 2,
    ) -> Dict[str, float]:
        """Initialize A/B from a truncated SVD of a [out, in] dense block."""
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape {tuple(weight.shape)} does not match "
                f"({self.out_features}, {self.in_features})"
            )
        source = weight.detach().to(device=self.A.weight.device, dtype=tc.float32)
        used = min(self.rank, min(self.in_features, self.out_features))
        resolved_method = str(method).strip().lower()
        if resolved_method == "auto":
            resolved_method = "full_svd" if used == min(self.in_features, self.out_features) else "randomized_svd"
        if resolved_method == "full_svd":
            u, s, vh = tc.linalg.svd(source, full_matrices=False)
        elif resolved_method == "randomized_svd":
            q = min(min(source.shape), used + max(0, int(oversampling)))
            u, s, v = tc.pca_lowrank(source, q=q, center=False, niter=max(0, int(niter)))
            vh = v.transpose(0, 1)
        else:
            raise ValueError(f"unsupported low-rank initialization method={method!r}")
        self.A.weight.zero_()
        self.B.weight.zero_()
        self.A.weight[:, :used].copy_((u[:, :used] * s[:used]).to(dtype=self.A.weight.dtype))
        self.B.weight[:used, :].copy_(vh[:used, :].to(dtype=self.B.weight.dtype))
        recon = self.A.weight.to(tc.float32) @ self.B.weight.to(tc.float32)
        denom = source.norm().clamp_min(1e-12)
        rel_error = float((source - recon).norm().div(denom).item())
        captured = float(1.0 - rel_error * rel_error)
        return {
            "used_rank": float(used),
            "method": resolved_method,
            "relative_frobenius_error": rel_error,
            "captured_energy": captured,
        }


class BlockPartitionedLinear(nn.Module):
    """Exact four-block linear map used to prove the split/add/concat algebra."""

    def __init__(self, in_features: int, out_features: int, main_in: int, main_out: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.main_in_features = int(main_in)
        self.main_out_features = int(main_out)
        self.tail_in_features = self.in_features - self.main_in_features
        self.tail_out_features = self.out_features - self.main_out_features
        if min(self.main_in_features, self.main_out_features, self.tail_in_features, self.tail_out_features) <= 0:
            raise ValueError("BlockPartitionedLinear requires non-empty main and tail dimensions")
        self.W00 = nn.Linear(self.main_in_features, self.main_out_features, bias=False)
        self.W01 = nn.Linear(self.tail_in_features, self.main_out_features, bias=False)
        self.W10 = nn.Linear(self.main_in_features, self.tail_out_features, bias=False)
        self.W11 = nn.Linear(self.tail_in_features, self.tail_out_features, bias=False)

    @tc.no_grad()
    def set_from_dense_weight(self, weight: tc.Tensor) -> None:
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(f"unexpected dense weight shape: {tuple(weight.shape)}")
        mo, mi = self.main_out_features, self.main_in_features
        self.W00.weight.copy_(weight[:mo, :mi])
        self.W01.weight.copy_(weight[:mo, mi:])
        self.W10.weight.copy_(weight[mo:, :mi])
        self.W11.weight.copy_(weight[mo:, mi:])

    def forward(self, x: tc.Tensor) -> tc.Tensor:
        x0, x1 = x[..., : self.main_in_features], x[..., self.main_in_features :]
        y0 = self.W00(x0) + self.W01(x1)
        y1 = self.W10(x0) + self.W11(x1)
        return tc.cat((y0, y1), dim=-1)

    def forward_with_trace(self, x: tc.Tensor, trace: TraceFn = print) -> tc.Tensor:
        trace(f"[BlockPartitionedLinear] input={tuple(x.shape)}")
        trace(
            f"[split] x0={self.main_in_features} x1={self.tail_in_features}; "
            f"output y0={self.main_out_features} y1={self.tail_out_features}"
        )
        y = self.forward(x)
        trace("[compute] y0=W00(x0)+W01(x1); y1=W10(x0)+W11(x1)")
        trace(f"[concat] output={tuple(y.shape)} padding=False crop=False")
        return y


class SparseMiXTLinear(nn.Module):
    """Power-of-d MiXT main block plus three explicit low-rank boundaries.

    Routing, experts, TopK/soft dispatch, global expert, q calculation, and
    k_out derivation are delegated to the existing GatedADTNLayer unchanged.
    Only padding/cropping is replaced by real-dimension block composition.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        cfg: Mapping[str, Any],
        ranks: Mapping[str, int],
        main_in_features: int = 0,
        main_out_features: int = 0,
        router_input_policy: str = "main",
        dtype: tc.dtype = tc.float32,
        device: Optional[tc.device] = None,
        init_stats: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.d = int(cfg.get("d", 2))
        self.main_in_features = resolve_main_dim(self.in_features, main_in_features, self.d)
        self.main_out_features = resolve_main_dim(self.out_features, main_out_features, self.d)
        self.tail_in_features = self.in_features - self.main_in_features
        self.tail_out_features = self.out_features - self.main_out_features
        self.router_input_policy = str(router_input_policy).strip().lower()
        if self.router_input_policy not in {"main", "full_real"}:
            raise ValueError(f"unsupported router_input_policy={router_input_policy!r}")

        backend_version = int(dict(cfg.get("sparse_mixt") or {}).get("backend_version", 1))
        if backend_version != 1:
            raise ValueError(f"unsupported sparse_mixt backend_version={backend_version}; expected 1")

        gate_dim = self.main_in_features if self.router_input_policy == "main" else self.in_features
        gate_cfg = gate_config_from_mapping(cfg, data_dim=gate_dim)
        self.core = GatedADTNLayer(
            dim_input=self.main_in_features,
            dim_output=self.main_out_features,
            num_blocks=int(cfg["E"]),
            d=self.d,
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
        )
        if self.router_input_policy == "full_real":
            full_gate_cfg = replace(gate_cfg, num_experts=self.core.num_blocks, data_dim=self.in_features)
            self.core.gate = gate_factory_config(full_gate_cfg).to(dtype=tc.float32, device=device)

        rank01 = int(ranks["01"])
        rank10 = int(ranks["10"])
        rank11 = int(ranks["11"])
        branch_dtype = cfg.get("dtype", dtype)
        self.lr01 = (
            LowRankLinear(self.tail_in_features, self.main_out_features, rank01, dtype=branch_dtype, device=device)
            if self.tail_in_features > 0
            else None
        )
        self.lr10 = (
            LowRankLinear(self.main_in_features, self.tail_out_features, rank10, dtype=branch_dtype, device=device)
            if self.tail_out_features > 0
            else None
        )
        self.lr11 = (
            LowRankLinear(self.tail_in_features, self.tail_out_features, rank11, dtype=branch_dtype, device=device)
            if self.tail_in_features > 0 and self.tail_out_features > 0
            else None
        )

        self.q_in = self.core.q_in
        self.q_out = self.core.q_out
        self.in_pad = self.in_features
        self.out_pad = self.out_features

    def _route(self, x: tc.Tensor, x0: tc.Tensor):
        gate_input = x0 if self.router_input_policy == "main" else x
        return self.core.gate(gate_input.to(dtype=tc.float32))

    def forward(self, x: tc.Tensor, **kwargs):
        if int(x.shape[-1]) != self.in_features:
            raise ValueError(f"expected input last dim {self.in_features}, got {int(x.shape[-1])}")
        orig_shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        x0, x1 = x2[:, : self.main_in_features], x2[:, self.main_in_features :]
        core_kwargs = dict(kwargs)
        if self.router_input_policy == "full_real" and not any(k in core_kwargs for k in ("probs", "mask", "idx_w")) and not core_kwargs.get("dense", False):
            probs, mask, aux = self._route(x2, x0)
            core_kwargs.update(probs=probs, mask=mask, aux=aux)
        main_out = self.core(x0, **core_kwargs)
        if isinstance(main_out, tuple):
            y00, aux = main_out
        else:
            y00, aux = main_out, None
        y0 = y00 if self.lr01 is None else y00 + self.lr01(x1)
        if self.tail_out_features > 0:
            if self.lr10 is None:
                raise RuntimeError("tail output requires lr10")
            y1 = self.lr10(x0)
            if self.lr11 is not None:
                y1 = y1 + self.lr11(x1)
            y2 = tc.cat((y0, y1), dim=-1)
        else:
            y2 = y0
        y = y2.reshape(*orig_shape[:-1], self.out_features)
        return y, aux

    def forward_with_trace(self, x: tc.Tensor, trace: TraceFn = print, **kwargs):
        trace(
            f"[SparseMiXTLinear] real={self.in_features}->{self.out_features} "
            f"main={self.main_in_features}->{self.main_out_features} "
            f"tail={self.tail_in_features}->{self.tail_out_features}"
        )
        trace(
            f"[MiXT] d={self.d} q_in={self.core.q_in} q_out={self.core.q_out} "
            f"k_in={self.core.k_in} k_out={self.core.k_out} "
            f"experts={self.core.num_blocks} router={self.router_input_policy}"
        )
        trace(
            f"[routing] gate={self.core.gate.__class__.__name__} "
            f"topk={getattr(self.core.gate, 'k', 'soft')} "
            f"global_expert={self.core.global_expert_enabled}"
        )
        y0_formula = "MiXT00(x0)" + ("+LR01(x1)" if self.lr01 is not None else "")
        if self.tail_out_features > 0:
            y1_formula = "LR10(x0)" + ("+LR11(x1)" if self.lr11 is not None else "")
            trace(f"[boundaries] y0={y0_formula}; y1={y1_formula}")
        else:
            trace(f"[boundaries] y={y0_formula}; output_tail=0")
        y, aux = self.forward(x, **kwargs)
        trace(f"[concat] output={tuple(y.shape)} padding=False crop=False")
        return y, aux

    @tc.no_grad()
    def initialize_from_dense_weight(
        self,
        weight: tc.Tensor,
        *,
        initialize_main_exact: bool = False,
        boundary_method: str = "auto",
        svd_oversampling: int = 8,
        svd_niter: int = 2,
    ) -> Dict[str, Any]:
        """Initialize boundary SVDs and optionally a single full-site main block."""
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(f"unexpected dense weight shape: {tuple(weight.shape)}")
        mo, mi = self.main_out_features, self.main_in_features
        report: Dict[str, Any] = {}
        if self.lr01 is not None:
            report["01"] = self.lr01.set_from_dense_weight(
                weight[:mo, mi:], method=boundary_method, oversampling=svd_oversampling, niter=svd_niter
            )
        if self.lr10 is not None:
            report["10"] = self.lr10.set_from_dense_weight(
                weight[mo:, :mi], method=boundary_method, oversampling=svd_oversampling, niter=svd_niter
            )
        if self.lr11 is not None:
            report["11"] = self.lr11.set_from_dense_weight(
                weight[mo:, mi:], method=boundary_method, oversampling=svd_oversampling, niter=svd_niter
            )
        if initialize_main_exact:
            if self.core.num_blocks != 1 or self.core.k_in != self.core.q_in:
                raise ValueError("exact main initialization requires one expert and k_in == q_in")
            main = weight[:mo, :mi].detach().to(device=self.core.blocks[0].U.device, dtype=self.core.blocks[0].U.dtype)
            self.core.blocks[0].U.copy_(main.transpose(0, 1).reshape_as(self.core.blocks[0].U))
            report["00"] = {"exact": True}
        return report
