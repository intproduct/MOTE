from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from ..ADTN import (
    BlockMeta,
    TensorBlock,
    apply_block_to_sites,
    block_init_stats_are_usable,
    dense_weight_init_stats,
    make_unique_positions,
    q_from_dim_pad,
)


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


class FixedADTNCore(nn.Module):
    def __init__(
        self,
        *,
        dim_input: int,
        dim_output: int,
        num_blocks: int,
        d: int,
        k_in: int,
        pos_strategy: str = "sliding",
        dtype: tc.dtype = tc.float32,
        device: Optional[tc.device] = None,
        init_gamma: float = 0.6,
        seed: int = 0,
        warmup_ratio: float = 0.5,
        warmup_stride: int = 1,
        block_init_mode: str = "gamma_normal",
        block_init_std_scale: float = 1.0,
        block_init_trunc_std: float = 2.0,
        init_stats: Optional[Dict[str, object]] = None,
    ):
        super().__init__()
        self.dim_input = int(dim_input)
        self.dim_output = int(dim_output)
        self.num_blocks = int(num_blocks)
        self.d = int(d)
        self.k_in = int(k_in)
        self.q_in = q_from_dim_pad(self.dim_input, self.d)
        self.q_out = q_from_dim_pad(self.dim_output, self.d)
        self.k_out = int(self.k_in + self.q_out - self.q_in)
        if self.k_out < 0:
            raise ValueError(
                f"[FixedADTN] k_out < 0: k_in={self.k_in}, q_in={self.q_in}, q_out={self.q_out}. "
                f"Choose larger k_in or change d/padding."
            )

        positions, n_eff = make_unique_positions(
            q_in=self.q_in,
            k_in=self.k_in,
            N=self.num_blocks,
            strategy=str(pos_strategy),
            seed=int(seed),
            warmup_ratio=float(warmup_ratio),
            warmup_stride=int(warmup_stride),
        )
        self.num_blocks = int(n_eff)
        self.positions: List[List[int]] = positions
        self.pos_stats = self._coverage_stats(self.positions, self.q_in)

        fan_in = (self.d ** self.k_in) if self.k_in > 0 else 1
        init_std = 1.0 / (fan_in ** float(init_gamma))
        self.blocks = nn.ModuleList(
            [
                TensorBlock(
                    meta=BlockMeta(block_id=i, pos_in=pos),
                    d=self.d,
                    k_in=self.k_in,
                    k_out=self.k_out,
                    dtype=dtype,
                    device=device,
                    init_std=init_std,
                    block_init_mode=str(block_init_mode),
                    block_init_std_scale=float(block_init_std_scale),
                    block_init_trunc_std=float(block_init_trunc_std),
                    init_stats=init_stats,
                    warn_on_init_fallback=(i == 0),
                )
                for i, pos in enumerate(self.positions)
            ]
        )

        self.enable_usage_tracking = True
        self.last_aux_dict = None
        self.last_usage = None
        self.last_top1 = None
        self.last_usage_counts = None
        self.last_top1_counts = None

    @staticmethod
    def _coverage_stats(positions: List[List[int]], q_in: int) -> Dict[str, Any]:
        cnt = tc.zeros(q_in, dtype=tc.int32, device="cpu")
        for pos in positions:
            for i in pos:
                cnt[i] += 1
        cnt_f = cnt.to(tc.float32)
        mean = float(cnt_f.mean().item()) if q_in > 0 else 0.0
        std = float(cnt_f.std(unbiased=False).item()) if q_in > 0 else 0.0
        cv = float(std / (mean + 1e-8))
        return {
            "q_in": q_in,
            "N": len(positions),
            "k_in": len(positions[0]) if positions else 0,
            "mean": mean,
            "std": std,
            "cv": cv,
            "min": int(cnt.min().item()) if q_in > 0 else 0,
            "max": int(cnt.max().item()) if q_in > 0 else 0,
            "cnt": cnt.tolist(),
        }

    def reshape_in(self, x: tc.Tensor) -> Tuple[tc.Tensor, int, int]:
        if x.ndim == 3:
            bsz, seqlen, hidden = x.shape
            x2 = x.reshape(bsz * seqlen, hidden)
            return x2.reshape(bsz * seqlen, *([self.d] * self.q_in)), bsz, seqlen
        if x.ndim == 2:
            n, hidden = x.shape
            return x.reshape(n, *([self.d] * self.q_in)), 0, 0
        raise ValueError(f"x must be [B,S,H] or [N,H], got {tuple(x.shape)}")

    def reshape_out(self, y_sites: tc.Tensor, bsz: int, seqlen: int) -> tc.Tensor:
        y = y_sites.reshape(y_sites.shape[0], self.dim_output)
        if bsz > 0:
            return y.reshape(bsz, seqlen, self.dim_output)
        return y

    def set_usage_tracking_enabled(self, enabled: bool) -> None:
        self.enable_usage_tracking = bool(enabled)
        if not self.enable_usage_tracking:
            self.reset_runtime_usage_cache()

    def reset_runtime_usage_cache(self) -> None:
        self.last_aux_dict = None
        self.last_usage = None
        self.last_top1 = None
        self.last_usage_counts = None
        self.last_top1_counts = None

    def collect_runtime_usage_tensors(self) -> Dict[str, Optional[tc.Tensor]]:
        return {
            "usage": self.last_usage,
            "top1": self.last_top1,
            "expert_counts": self.last_usage_counts,
            "top1_counts": self.last_top1_counts,
            "importance": None,
            "load": None,
            "drop_rate": None,
            "capacity": None,
            "entropy_soft": None,
            "entropy_hard": None,
            "aux": None,
        }

    def materialize_usage_report(self) -> Dict[str, Optional[object]]:
        return {
            "usage": None if self.last_usage is None else self.last_usage.detach().cpu().tolist(),
            "top1": None if self.last_top1 is None else self.last_top1.detach().cpu().tolist(),
            "entropy": None,
            "load_balance": None,
            "active_expert_count": None,
            "max_expert_share": None,
            "expert_cv": None,
            "importance": None,
            "load": None,
            "drop_rate": None,
            "capacity": None,
            "entropy_soft_token": None,
            "entropy_soft_batch": None,
            "entropy_hard_token": None,
            "entropy_hard_batch": None,
            "global_expert_enabled": False,
            "global_pos_in": None,
        }

    def _update_usage(self, probs: Optional[tc.Tensor], num_tokens: int) -> None:
        if not self.enable_usage_tracking:
            return
        if probs is None:
            counts = tc.full(
                (self.num_blocks,),
                fill_value=float(num_tokens) / float(max(1, self.num_blocks)),
                dtype=tc.float32,
            )
            usage = counts / max(1, num_tokens)
        else:
            probs_fp32 = probs.detach().to(tc.float32)
            counts = probs_fp32.sum(dim=0)
            usage = probs_fp32.mean(dim=0)
        self.last_usage_counts = counts
        self.last_usage = usage
        self.last_top1_counts = None
        self.last_top1 = None

    def forward(
        self,
        x: tc.Tensor,
        return_aux: bool = False,
        *,
        probs: Optional[tc.Tensor] = None,
        mask: Optional[tc.Tensor] = None,
        aux: Optional[Dict[str, Any]] = None,
        idx_w: Optional[Tuple[tc.Tensor, tc.Tensor]] = None,
        dense: bool = False,
        use_global_expert: Optional[bool] = None,
    ):
        del mask, aux, idx_w, use_global_expert
        x = x.to(dtype=tc.float32)
        x_sites, bsz, seqlen = self.reshape_in(x)
        num_tokens = x_sites.shape[0]
        probs_2d: Optional[tc.Tensor] = None
        if probs is not None:
            probs_2d = probs.reshape(num_tokens, self.num_blocks).to(device=x_sites.device, dtype=tc.float32)
        elif dense:
            probs_2d = None

        self._update_usage(probs_2d, num_tokens)

        acc = None
        for idx, block in enumerate(self.blocks):
            yb = apply_block_to_sites(
                x_sites=x_sites,
                U=block.U,
                pos_in=self.positions[idx],
                d=self.d,
                q_in=self.q_in,
                k_in=self.k_in,
                k_out=self.k_out,
            )
            if probs_2d is not None:
                wb = probs_2d[:, idx].view(num_tokens, *([1] * (yb.ndim - 1)))
                yb = yb * wb
            acc = yb if acc is None else (acc + yb)
        if acc is None:
            raise RuntimeError("FixedADTNCore has no blocks")
        if probs_2d is None:
            acc = acc / float(self.num_blocks)
        y = self.reshape_out(acc, bsz, seqlen)
        if return_aux:
            return y, None
        return y


class FixedADTNLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        cfg: Dict[str, Any],
        device: tc.device,
        dtype: tc.dtype = tc.float32,
        init_stats: Optional[Dict[str, object]] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.d = int(cfg.get("d", 2))
        self.q_in = q_from_dim_pad(self.in_features, self.d)
        self.q_out = q_from_dim_pad(self.out_features, self.d)
        self.in_pad = self.d ** self.q_in
        self.out_pad = self.d ** self.q_out
        self.core = FixedADTNCore(
            dim_input=self.in_pad,
            dim_output=self.out_pad,
            num_blocks=int(cfg["E"]),
            d=self.d,
            k_in=int(cfg.get("k_in", 3)),
            pos_strategy=str(cfg.get("pos_strategy", "random")),
            dtype=cfg.get("dtype", dtype),
            device=device,
            init_gamma=float(cfg.get("init_gamma", 0.6)),
            seed=int(cfg.get("seed", 0)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.5)),
            warmup_stride=int(cfg.get("warmup_stride", 1)),
            block_init_mode=str(cfg.get("block_init_mode", "gamma_normal")),
            block_init_std_scale=float(cfg.get("block_init_std_scale", 1.0)),
            block_init_trunc_std=float(cfg.get("block_init_trunc_std", 2.0)),
            init_stats=init_stats,
        )

    def forward(self, x: tc.Tensor, **kwargs):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        if self.in_pad != self.in_features:
            x2 = F.pad(x2, (0, self.in_pad - self.in_features))
        out = self.core(x2, **kwargs)
        y2 = unwrap_y(out).reshape(x2.shape[0], -1)
        if self.out_pad != self.out_features:
            y2 = y2[:, : self.out_features]
        y = y2.reshape(*orig_shape[:-1], self.out_features)
        if isinstance(out, tuple):
            return y, out[1]
        return y


class ADTNBaselineFFNLayer(nn.Module):
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
                _log_projection_block_init(log, layer_idx=layer_idx, proj_name=proj_name, backend="adtn_fixed", cfg=cfg, stats=stats)
        self.gate_proj = FixedADTNLinear(
            self.hidden_size,
            self.intermediate_size,
            cfg=cfg,
            device=device,
            dtype=dtype,
            init_stats=None if init_stats is None else init_stats["gate_proj"],
        ).to(device)
        self.up_proj = FixedADTNLinear(
            self.hidden_size,
            self.intermediate_size,
            cfg=cfg,
            device=device,
            dtype=dtype,
            init_stats=None if init_stats is None else init_stats["up_proj"],
        ).to(device)
        down_cfg = dict(cfg)
        down_cfg["k_in"] = int(self.gate_proj.core.k_out)
        self.down_proj = FixedADTNLinear(
            self.intermediate_size,
            self.hidden_size,
            cfg=down_cfg,
            device=device,
            dtype=dtype,
            init_stats=None if init_stats is None else init_stats["down_proj"],
        ).to(device)
        self.layer_idx = int(layer_idx)
        self.fitmotn_block_layout: Dict[str, Dict[str, Any]] = {}
        self.fitmotn_expert_warmup_state = {"enabled": False}

    def _build_fixed_probs(self, proj: FixedADTNLinear, proj_name: str, batch_size: int, device: tc.device) -> Optional[tc.Tensor]:
        warmup_state = self.fitmotn_expert_warmup_state if isinstance(self.fitmotn_expert_warmup_state, dict) else {}
        proj_state = warmup_state.get(proj_name) if isinstance(warmup_state.get(proj_name), dict) else None
        enabled = bool(warmup_state.get("enabled")) and self.training and proj_state is not None
        if not enabled:
            return None
        num_blocks = int(proj.core.num_blocks)
        probs = tc.ones((int(batch_size), num_blocks), dtype=tc.float32, device=device)
        random_indices = list(proj_state.get("random_indices", []))
        random_scale = float(proj_state.get("random_scale", 1.0))
        if random_indices:
            ridx = tc.tensor(sorted(set(int(i) for i in random_indices)), dtype=tc.long, device=device)
            probs[:, ridx] *= random_scale
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return probs / denom

    def _forward_proj(self, proj: FixedADTNLinear, x: tc.Tensor, proj_name: str) -> tc.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]).to(dtype=tc.float32)
        probs = self._build_fixed_probs(proj, proj_name, x2.shape[0], x2.device)
        if probs is None:
            return unwrap_y(proj(x))
        if proj.in_pad != proj.in_features:
            x2 = F.pad(x2, (0, proj.in_pad - proj.in_features))
        out = proj.core(x2, probs=probs, mask=None, aux=None)
        y2 = unwrap_y(out).reshape(x2.shape[0], -1)
        if proj.out_pad != proj.out_features:
            y2 = y2[:, : proj.out_features]
        return y2.reshape(*orig_shape[:-1], proj.out_features)

    def forward(self, x: tc.Tensor) -> tc.Tensor:
        in_dtype = x.dtype
        g = self._forward_proj(self.gate_proj, x, "gate_proj")
        u = self._forward_proj(self.up_proj, x, "up_proj")
        h = self.act(g) * u
        y = self._forward_proj(self.down_proj, h, "down_proj")
        return y.to(dtype=in_dtype)
