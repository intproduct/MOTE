from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch as tc
import torch.nn.functional as F

from ..ADTN import apply_block_to_sites
from ..model import unwrap_y
from .spectral import compare_matrices, spectral_stats

PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


def _flat(x: tc.Tensor, max_tokens: int | None = None) -> tc.Tensor:
    y = x.reshape(-1, x.shape[-1])
    if max_tokens is not None and max_tokens > 0:
        y = y[: int(max_tokens)]
    return y


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_npy(path: Path, tensor: tc.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = tensor.detach().cpu()
    if t.dtype is tc.bfloat16:
        t = t.float()
    np.save(path, t.numpy())


def _first_param_device_dtype(module) -> tuple[tc.device, tc.dtype | None]:
    try:
        param = next(module.parameters())
        return param.device, param.dtype
    except StopIteration:
        return tc.device("cpu"), None


def move_to_module_device_dtype(x: tc.Tensor, module) -> tc.Tensor:
    param = next(module.parameters(), None)
    if param is None:
        return x
    return x.to(device=param.device, dtype=param.dtype)


def gini(values: tc.Tensor) -> float:
    x = values.detach().to(device="cpu", dtype=tc.float32).flatten()
    if x.numel() == 0:
        return 0.0
    if float(x.sum().item()) <= 0.0:
        return 0.0
    x, _ = tc.sort(x)
    n = x.numel()
    idx = tc.arange(1, n + 1, dtype=tc.float32, device=x.device)
    return float(((2 * idx - n - 1) * x).sum().item() / (n * x.sum().item()))


def _pad_projection_input(proj, x: tc.Tensor) -> tc.Tensor:
    param_device, param_dtype = _first_param_device_dtype(proj)
    x2 = _flat(x).to(device=param_device, dtype=param_dtype or x.dtype)
    if int(proj.in_pad) != int(proj.in_features):
        x2 = F.pad(x2, (0, int(proj.in_pad) - int(proj.in_features)))
    return x2


def _selected_ids(probs: tc.Tensor, mask: Optional[tc.Tensor], k: int) -> tuple[tc.Tensor, tc.Tensor]:
    if mask is not None:
        selected = probs.masked_fill(mask <= 0, float("-inf"))
        kk = max(1, min(int(k), probs.shape[-1]))
        top_probs, top_ids = tc.topk(selected, k=kk, dim=-1)
        top_probs = tc.where(tc.isfinite(top_probs), top_probs, tc.zeros_like(top_probs))
        return top_ids, top_probs
    kk = max(1, min(int(k), probs.shape[-1]))
    top_probs, top_ids = tc.topk(probs, k=kk, dim=-1)
    return top_ids, top_probs


def _matrix_stats(y: tc.Tensor, top_k_svd: int) -> Dict[str, Any]:
    return spectral_stats(y, top_k_svd=top_k_svd)


def trace_motn_projection(
    proj,
    x_proj: tc.Tensor,
    *,
    max_tokens: int = 4096,
    max_block_tokens: int = 1024,
    max_blocks_trace: str | int = "all",
    top_k_svd: int = 128,
    save_router_tensors: bool = False,
    out_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    core = getattr(proj, "core", None)
    gate = getattr(core, "gate", None)
    router = getattr(gate, "router", None)
    if core is None or gate is None or router is None:
        return {"available": False, "warning": "projection is not a MoTNLayer with core.gate.router"}

    x_padded = _pad_projection_input(proj, _flat(x_proj, max_tokens=max_tokens))
    with tc.no_grad():
        logits = router(x_padded).detach()
        probs, mask, aux = gate(x_padded)
        probs = probs.detach()
        mask_det = None if mask is None else mask.detach()
        if probs.ndim == 3:
            probs = probs.reshape(-1, probs.shape[-1])
        if mask_det is not None and mask_det.ndim == 3:
            mask_det = mask_det.reshape(-1, mask_det.shape[-1])

        k = int(getattr(gate, "k", min(1, probs.shape[-1])) or min(1, probs.shape[-1]))
        topk_ids, topk_probs = _selected_ids(probs, mask_det, k)
        top1_ids = topk_ids[:, 0] if topk_ids.numel() else tc.empty(0, device=probs.device, dtype=tc.long)
        p = probs.float().clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=-1)
        if mask_det is not None:
            usage_counts = mask_det.float().sum(dim=0)
        else:
            usage_counts = probs.float().sum(dim=0)
        usage_total = usage_counts.sum().clamp_min(1e-12)
        usage_ratio = usage_counts / usage_total

        x_block = x_padded[: int(max_block_tokens)] if int(max_block_tokens) > 0 else x_padded
        probs_block = probs[: x_block.shape[0]]
        mask_block = None if mask_det is None else mask_det[: x_block.shape[0]]
        top1_block = top1_ids[: x_block.shape[0]]
        x_sites, _, _ = core.reshape_in(x_block)
        num_blocks = int(core.num_blocks)
        if str(max_blocks_trace).lower() == "all":
            trace_blocks = num_blocks
        else:
            trace_blocks = max(0, min(num_blocks, int(max_blocks_trace)))
        skipped_blocks = list(range(trace_blocks, num_blocks))
        block_stats: Dict[str, Any] = {}
        routed_acc = None
        pair_means = []
        block_outputs = []
        for b in range(trace_blocks):
            yb_sites = apply_block_to_sites(
                x_sites,
                core.blocks[b].U,
                core.positions[b],
                d=core.d,
                q_in=core.q_in,
                k_in=core.k_in,
                k_out=core.k_out,
            )
            yb = yb_sites.reshape(yb_sites.shape[0], int(core.dim_output))
            if int(proj.out_pad) != int(proj.out_features):
                yb = yb[:, : int(proj.out_features)]
            norms = yb.float().norm(dim=-1)
            selected_mask = (mask_block[:, b] > 0) if mask_block is not None else (top1_block == b)
            sel = yb[selected_mask]
            yb_stats = _matrix_stats(yb, top_k_svd)
            sel_stats = _matrix_stats(sel, top_k_svd) if sel.numel() else None
            block_stats[str(b)] = {
                "all": {
                    "norm_mean": float(norms.mean().item()) if norms.numel() else 0.0,
                    "norm_std": float(norms.std(unbiased=False).item()) if norms.numel() > 1 else 0.0,
                    "effective_rank": yb_stats["effective_rank"],
                    "cov_trace": yb_stats["covariance_trace"],
                },
                "selected": {
                    "count": int(selected_mask.sum().item()),
                    "norm_mean": float(sel.float().norm(dim=-1).mean().item()) if sel.numel() else 0.0,
                    "norm_std": float(sel.float().norm(dim=-1).std(unbiased=False).item()) if sel.shape[0] > 1 else 0.0,
                    "effective_rank": sel_stats["effective_rank"] if sel_stats is not None else 0.0,
                    "cov_trace": sel_stats["covariance_trace"] if sel_stats is not None else 0.0,
                },
            }
            wb = probs_block[:, b].view(-1, 1).to(yb.dtype)
            if mask_block is not None:
                wb = wb * mask_block[:, b].view(-1, 1).to(yb.dtype)
            contrib = yb * wb
            routed_acc = contrib if routed_acc is None else routed_acc + contrib
            block_outputs.append(yb.detach().float().cpu())
        for i in range(len(block_outputs)):
            row = []
            for j in range(len(block_outputs)):
                row.append(float(F.cosine_similarity(block_outputs[i], block_outputs[j], dim=-1).mean().item()))
            pair_means.append(row)

        global_stats = None
        if bool(getattr(core, "global_expert_enabled", False)) and getattr(core, "global_block", None) is not None:
            gy_sites = core._apply_global_expert(x_sites)
            gy = gy_sites.reshape(gy_sites.shape[0], int(core.dim_output))
            if int(proj.out_pad) != int(proj.out_features):
                gy = gy[:, : int(proj.out_features)]
            gnorm = gy.float().norm(dim=-1)
            rnorm = routed_acc.float().norm(dim=-1) if routed_acc is not None else tc.zeros_like(gnorm)
            global_stats = {
                "global_output_norm_mean": float(gnorm.mean().item()) if gnorm.numel() else 0.0,
                "global_output_norm_std": float(gnorm.std(unbiased=False).item()) if gnorm.numel() > 1 else 0.0,
                "global_output_cov_trace": _matrix_stats(gy, top_k_svd)["covariance_trace"],
                "global_vs_routed_output_cosine": float(F.cosine_similarity(gy.float(), routed_acc.float(), dim=-1).mean().item()) if routed_acc is not None else None,
                "global_output_routed_output_norm_ratio": float((gnorm.mean() / rnorm.mean().clamp_min(1e-12)).item()) if rnorm.numel() else None,
            }

    router_stats = {
        "available": True,
        "logits_mean": float(logits.float().mean().item()) if logits.numel() else 0.0,
        "logits_std": float(logits.float().std(unbiased=False).item()) if logits.numel() > 1 else 0.0,
        "entropy_mean": float(entropy.mean().item()) if entropy.numel() else 0.0,
        "entropy_std": float(entropy.std(unbiased=False).item()) if entropy.numel() > 1 else 0.0,
        "expert_usage_counts": [float(v) for v in usage_counts.detach().cpu().tolist()],
        "expert_usage_ratio": [float(v) for v in usage_ratio.detach().cpu().tolist()],
        "usage_gini": gini(usage_counts),
        "active_expert_count": int((usage_counts > 0).sum().item()),
    }

    if out_paths is not None:
        routing_dir = Path(out_paths.get("routing_dir", "."))
        prefix = str(out_paths.get("prefix", "projection"))
        _json_dump(routing_dir / f"{prefix}_logits_stats.json", router_stats)
        if save_router_tensors:
            _save_npy(routing_dir / f"{prefix}_topk_ids.npy", topk_ids.long())
            _save_npy(routing_dir / f"{prefix}_top1_ids.npy", top1_ids.long())
            _save_npy(routing_dir / f"{prefix}_logits.npy", logits)
            _save_npy(routing_dir / f"{prefix}_probs.npy", probs)
            if mask_det is not None:
                _save_npy(routing_dir / f"{prefix}_mask.npy", mask_det.float())
            _save_npy(routing_dir / f"{prefix}_topk_probs.npy", topk_probs)

    return {
        "router": router_stats,
        "blocks": {
            "per_block": block_stats,
            "pairwise_cosine_mean_matrix": pair_means,
            "diversity_summary": {
                "tokens_used": int(x_block.shape[0]),
                "blocks_traced": int(trace_blocks),
                "blocks_total": int(num_blocks),
                "skipped_blocks": skipped_blocks,
                "mean_pairwise_cosine": float(np.mean(pair_means)) if pair_means else 0.0,
                "mean_selected_count": float(np.mean([v["selected"]["count"] for v in block_stats.values()])) if block_stats else 0.0,
            },
        },
        "global": global_stats,
    }


def _proj_record(dense_in, dense_out, mote_in, mote_out, *, top_k_svd: int) -> Dict[str, Any]:
    return {
        "dense": {"input": spectral_stats(dense_in, top_k_svd=top_k_svd), "output": spectral_stats(dense_out, top_k_svd=top_k_svd)},
        "mote": {"input": spectral_stats(mote_in, top_k_svd=top_k_svd), "output": spectral_stats(mote_out, top_k_svd=top_k_svd)},
        "compare": compare_matrices(dense_out, mote_out, top_k_svd=top_k_svd),
    }


def trace_qwen_mlp_projection(
    base_mlp,
    mote_mlp,
    x: tc.Tensor,
    *,
    max_tokens: int = 4096,
    max_block_tokens: int = 1024,
    max_blocks_trace: str | int = "all",
    top_k_svd: int = 128,
    save_raw_tensors: bool = False,
    save_router_tensors: bool = False,
    out_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    x_flat = _flat(x, max_tokens=max_tokens)
    x_dense = move_to_module_device_dtype(x_flat, base_mlp)
    x_mote = move_to_module_device_dtype(x_flat, mote_mlp)
    with tc.no_grad():
        dense_gate = base_mlp.gate_proj(x_dense)
        dense_up = base_mlp.up_proj(x_dense)
        dense_act = getattr(base_mlp, "act_fn", None) or getattr(base_mlp, "act", None)
        dense_mid = dense_act(dense_gate) * dense_up
        dense_out = base_mlp.down_proj(dense_mid)

        mote_gate = unwrap_y(mote_mlp.gate_proj(x_mote))
        mote_up = unwrap_y(mote_mlp.up_proj(x_mote))
        mote_act = getattr(mote_mlp, "act", None) or getattr(mote_mlp, "act_fn", None)
        mote_mid = mote_act(mote_gate) * mote_up
        mote_out = unwrap_y(mote_mlp.down_proj(mote_mid))
        dense_mid_for_mote = move_to_module_device_dtype(dense_mid, mote_mlp.down_proj)
        mote_down_on_dense_mid = unwrap_y(mote_mlp.down_proj(dense_mid_for_mote))

    result: Dict[str, Any] = {}
    specs = {
        "gate_proj": (x_dense, dense_gate, x_mote, mote_gate),
        "up_proj": (x_dense, dense_up, x_mote, mote_up),
        "down_proj": (dense_mid, dense_out, mote_mid, mote_out),
    }
    for name, (din, dout, minp, mout) in specs.items():
        rec = _proj_record(din, dout, minp, mout, top_k_svd=top_k_svd)
        proj = getattr(mote_mlp, name, None)
        if proj is not None:
            trace = trace_motn_projection(
                proj,
                minp,
                max_tokens=max_tokens,
                max_block_tokens=max_block_tokens,
                max_blocks_trace=max_blocks_trace,
                top_k_svd=top_k_svd,
                save_router_tensors=save_router_tensors,
                out_paths=None if out_paths is None else {**out_paths, "prefix": f"{out_paths.get('prefix', 'layer')}_{name}"},
            )
            rec.update(trace)
        result[name] = rec
        if out_paths is not None:
            spectra_dir = Path(out_paths.get("spectra_dir", "."))
            spectra_dir.mkdir(parents=True, exist_ok=True)
            prefix = str(out_paths.get("prefix", "layer"))
            np.save(spectra_dir / f"{prefix}_{name}_dense_svals.npy", np.asarray(rec["dense"]["output"]["singular_values_topk"], dtype=np.float32))
            np.save(spectra_dir / f"{prefix}_{name}_mote_svals.npy", np.asarray(rec["mote"]["output"]["singular_values_topk"], dtype=np.float32))
            if save_raw_tensors:
                raw_dir = Path(out_paths.get("raw_dir", spectra_dir))
                _save_npy(raw_dir / f"{prefix}_{name}_dense_output.npy", dout)
                _save_npy(raw_dir / f"{prefix}_{name}_mote_output.npy", mout)

    result["mlp_output"] = {
        "dense": {"output": spectral_stats(dense_out, top_k_svd=top_k_svd)},
        "mote": {"output": spectral_stats(mote_out, top_k_svd=top_k_svd)},
        "compare": compare_matrices(dense_out, mote_out, top_k_svd=top_k_svd),
    }
    result["down_proj_shared_mid"] = {
        "dense": {
            "input": spectral_stats(dense_mid, top_k_svd=top_k_svd),
            "output": spectral_stats(dense_out, top_k_svd=top_k_svd),
        },
        "mote": {
            "input": spectral_stats(dense_mid_for_mote, top_k_svd=top_k_svd),
            "output": spectral_stats(mote_down_on_dense_mid, top_k_svd=top_k_svd),
        },
        "compare": compare_matrices(dense_out, mote_down_on_dense_mid, top_k_svd=top_k_svd),
    }
    return result
