from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import torch as tc
import torch.nn as nn

from ..audit import json_dump, jsonl_append, to_jsonable
from ..patching import PROJ_NAMES, iter_patched_motn_layers, resolve_operator_block_layout


def collect_dense_ffn_targets(model, layer_idxs) -> dict:
    targets = {}
    for layer_idx in sorted(set(int(i) for i in layer_idxs)):
        mlp = model.model.layers[layer_idx].mlp
        targets[layer_idx] = {
            "gate_proj": mlp.gate_proj.weight.detach().cpu().float().clone(),
            "up_proj": mlp.up_proj.weight.detach().cpu().float().clone(),
            "down_proj": mlp.down_proj.weight.detach().cpu().float().clone(),
        }
    return targets


def resolve_warmup_block_subsets(motn_operator) -> dict:
    layout = getattr(motn_operator, "fitmotn_block_layout", None) or getattr(motn_operator, "_fitmotn_block_layout", None)
    if not isinstance(layout, dict):
        layout = resolve_operator_block_layout(motn_operator)
    return {
        "n_total": int(layout.get("n_total", getattr(motn_operator.core, "num_blocks", 0))),
        "n_slide": int(layout.get("n_slide", 0)),
        "n_random": int(layout.get("n_random", 0)),
        "slide_indices": list(layout.get("slide_indices", [])),
        "random_indices": list(layout.get("random_indices", [])),
    }


def build_identity_operator_loader(in_dim, target_weight, batch_size, chunk_size):
    in_dim = int(in_dim)
    batch_size = max(1, int(batch_size))
    chunk_size = max(1, int(chunk_size))
    weight_t = target_weight.detach().cpu().float().t().contiguous()

    def _iterator() -> Iterator[Tuple[tc.Tensor, tc.Tensor]]:
        for start in range(0, in_dim, chunk_size):
            stop = min(in_dim, start + chunk_size)
            eye_chunk = tc.eye(stop - start, in_dim, dtype=tc.float32)
            for batch_start in range(0, eye_chunk.shape[0], batch_size):
                x = eye_chunk[batch_start : batch_start + batch_size].contiguous()
                y = x @ weight_t
                yield x, y

    return _iterator()


@tc.no_grad()
def _cosine_sim(a: tc.Tensor, b: tc.Tensor, eps: float = 1e-8) -> float:
    a = a.float()
    b = b.float()
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
    return float((a * b).sum(dim=-1).mean().item())


def _subset_probs(batch_size: int, num_blocks: int, active_block_indices: List[int], device: tc.device) -> tc.Tensor:
    probs = tc.zeros((int(batch_size), int(num_blocks)), dtype=tc.float32, device=device)
    if not active_block_indices:
        raise ValueError("active_block_indices is empty")
    active = tc.tensor(sorted(set(int(i) for i in active_block_indices)), dtype=tc.long, device=device)
    probs[:, active] = 1.0 / float(active.numel())
    return probs


@tc.no_grad()
def _reset_random_subset_scale(motn_operator, inactive_block_indices: List[int], scale: float) -> None:
    scale = float(scale)
    if math.isclose(scale, 1.0):
        return
    for idx in inactive_block_indices:
        block = motn_operator.core.blocks[int(idx)]
        for param in block.parameters():
            param.mul_(scale)


def _configure_trainable_subset(motn_operator, active_block_indices: List[int]) -> List[nn.Parameter]:
    active_set = {int(i) for i in active_block_indices}
    params: List[nn.Parameter] = []
    gate = getattr(motn_operator.core, "gate", None)
    if gate is not None:
        for p in gate.parameters():
            p.requires_grad_(False)
    for idx, block in enumerate(motn_operator.core.blocks):
        enabled = idx in active_set
        for p in block.parameters():
            p.requires_grad_(enabled)
            if enabled:
                params.append(p)
    return params


def _evaluate_operator(motn_operator, target_weight: tc.Tensor, active_block_indices: List[int], batch_size: int, chunk_size: int) -> Dict[str, float]:
    device = next(motn_operator.parameters()).device
    loader = build_identity_operator_loader(target_weight.shape[1], target_weight, batch_size=batch_size, chunk_size=chunk_size)
    total_sq = 0.0
    total_count = 0
    y_norm_sq = 0.0
    y_all: List[tc.Tensor] = []
    yhat_all: List[tc.Tensor] = []
    for x_cpu, y_cpu in loader:
        x = x_cpu.to(device=device, dtype=tc.float32)
        y = y_cpu.to(device=device, dtype=tc.float32)
        probs = _subset_probs(x.shape[0], motn_operator.core.num_blocks, active_block_indices, device=device)
        yhat = motn_operator(x, probs=probs, mask=None)
        if isinstance(yhat, tuple):
            yhat = yhat[0]
        err = yhat.float() - y.float()
        total_sq += float(err.pow(2).sum().item())
        total_count += int(err.numel())
        y_norm_sq += float(y.float().pow(2).sum().item())
        y_all.append(y.detach().cpu())
        yhat_all.append(yhat.detach().cpu())
    y_cat = tc.cat(y_all, dim=0)
    yhat_cat = tc.cat(yhat_all, dim=0)
    mse = total_sq / max(1, total_count)
    rmse = math.sqrt(mse)
    rel_l2 = math.sqrt(total_sq) / max(1e-8, math.sqrt(y_norm_sq))
    nrmse = rmse / max(1e-8, float(y_cat.float().std().item()))
    cos = _cosine_sim(yhat_cat, y_cat)
    return {"eval_mse": mse, "rel_l2": rel_l2, "nrmse": nrmse, "cos": cos}


def fit_single_motn_operator_subset(motn_operator, target_weight, active_block_indices, cfg, logger) -> dict:
    active_block_indices = sorted(set(int(i) for i in active_block_indices))
    subset_info = resolve_warmup_block_subsets(motn_operator)
    inactive_block_indices = [idx for idx in range(int(motn_operator.core.num_blocks)) if idx not in active_block_indices]
    _reset_random_subset_scale(motn_operator, inactive_block_indices, getattr(cfg, "init_random_std_scale", 1.0))
    params = _configure_trainable_subset(motn_operator, active_block_indices)
    if not params:
        raise ValueError("no trainable parameters found for active_block_indices")

    device = next(motn_operator.parameters()).device
    target_weight = target_weight.detach().cpu().float()
    optimizer = tc.optim.AdamW(params, lr=float(cfg.lr))
    loss_fn = nn.MSELoss()
    best_mse = None
    last_mse = None
    best_step = 0
    bad_steps = 0
    stop_reason = "max_steps"
    t0 = time.time()

    step_count = 0
    while step_count < int(cfg.steps_per_proj):
        loader = build_identity_operator_loader(
            target_weight.shape[1],
            target_weight,
            batch_size=int(cfg.batch_size),
            chunk_size=int(cfg.identity_chunk_size),
        )
        for x_cpu, y_cpu in loader:
            if step_count >= int(cfg.steps_per_proj):
                break
            step_count += 1
            x = x_cpu.to(device=device, dtype=tc.float32)
            y = y_cpu.to(device=device, dtype=tc.float32)
            probs = _subset_probs(x.shape[0], motn_operator.core.num_blocks, active_block_indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            yhat = motn_operator(x, probs=probs, mask=None)
            if isinstance(yhat, tuple):
                yhat = yhat[0]
            loss = loss_fn(yhat.float(), y.float())
            loss.backward()
            optimizer.step()

            last_mse = float(loss.detach().cpu().item())
            if best_mse is None or last_mse < best_mse - float(cfg.early_stop_min_delta):
                best_mse = last_mse
                best_step = step_count
                bad_steps = 0
            else:
                bad_steps += 1
                if bad_steps >= int(cfg.early_stop_patience):
                    stop_reason = "early_stop_patience"
                    break
            if cfg.target_rel_l2 is not None or cfg.target_cos is not None:
                eval_stats = _evaluate_operator(
                    motn_operator,
                    target_weight,
                    active_block_indices,
                    batch_size=int(cfg.batch_size),
                    chunk_size=int(cfg.identity_chunk_size),
                )
                if cfg.target_rel_l2 is not None and eval_stats["rel_l2"] <= float(cfg.target_rel_l2):
                    stop_reason = "target_rel_l2"
                    break
                if cfg.target_cos is not None and eval_stats["cos"] >= float(cfg.target_cos):
                    stop_reason = "target_cos"
                    break
        if stop_reason != "max_steps":
            break
    if step_count >= int(cfg.steps_per_proj) and stop_reason == "max_steps":
        stop_reason = "max_steps"

    eval_stats = _evaluate_operator(
        motn_operator,
        target_weight,
        active_block_indices,
        batch_size=int(cfg.batch_size),
        chunk_size=int(cfg.identity_chunk_size),
    )
    usage = getattr(motn_operator.core, "last_usage", None)
    result = {
        "train_mse_last": last_mse,
        "train_mse_best": best_mse,
        "best_step": best_step,
        "step_count": int(step_count),
        "stop_reason": stop_reason,
        "elapsed_sec": max(0.0, time.time() - t0),
        "subset": {
            "active_block_indices": active_block_indices,
            "inactive_block_indices": inactive_block_indices,
            "n_active": len(active_block_indices),
            "n_inactive": len(inactive_block_indices),
            "warmup": subset_info,
        },
        "usage": None if usage is None else usage.detach().cpu().tolist(),
    }
    result.update(eval_stats)
    if logger is not None:
        logger.info(
            "[Approx] proj blocks=%s active=%s mse=%.6e rel_l2=%.6e cos=%.6f stop=%s",
            motn_operator.core.num_blocks,
            len(active_block_indices),
            result["eval_mse"],
            result["rel_l2"],
            result["cos"],
            result["stop_reason"],
        )
    return result


def fit_patched_ffn_layer(motn_ffn_layer, dense_targets_for_this_layer, cfg, logger) -> dict:
    layer_result = {"layer_idx": int(getattr(motn_ffn_layer, "layer_idx", -1)), "projections": {}, "success": [], "failed": []}
    for proj_name in PROJ_NAMES:
        proj = getattr(motn_ffn_layer, proj_name)
        subsets = resolve_warmup_block_subsets(proj)
        active_indices = subsets["slide_indices"] if str(cfg.subset_mode) == "warmup_sliding_only" else list(range(subsets["n_total"]))
        try:
            result = fit_single_motn_operator_subset(
                proj,
                dense_targets_for_this_layer[proj_name],
                active_indices,
                cfg,
                logger,
            )
            layer_result["projections"][proj_name] = result
            layer_result["success"].append(proj_name)
        except Exception as exc:
            layer_result["projections"][proj_name] = {"error": repr(exc), "subset": subsets}
            layer_result["failed"].append(proj_name)
            if logger is not None:
                logger.exception("[Approx] layer=%s proj=%s failed", getattr(motn_ffn_layer, "layer_idx", -1), proj_name)
    return layer_result


def run_approx_init(model, layer_idxs, dense_targets, cfg, logger, run_dir) -> dict:
    run_dir = Path(run_dir)
    metrics_path = run_dir / str(cfg.metrics_jsonl_name)
    summary_path = run_dir / str(cfg.summary_json_name)
    summary = {
        "enabled": bool(cfg.enabled),
        "mode": str(cfg.mode),
        "subset_mode": str(cfg.subset_mode),
        "layers_requested": list(sorted(set(int(i) for i in layer_idxs))),
        "successes": [],
        "failures": [],
        "skipped": [],
        "layers": [],
        "started_at": time.time(),
    }
    effective_mode = str(cfg.mode).lower()
    if effective_mode != "identity":
        summary["requested_mode"] = effective_mode
        summary["effective_mode"] = "identity"
        summary["skipped"].append({"reason": f"approx_init mode '{cfg.mode}' is not fully implemented; fallback to identity"})
        if logger is not None:
            logger.warning("[Approx] mode=%s is not fully implemented; fallback to identity operator fitting", cfg.mode)
    else:
        summary["effective_mode"] = "identity"

    layer_map = {int(idx): module for idx, module in iter_patched_motn_layers(model)}
    for layer_idx in summary["layers_requested"]:
        layer = layer_map.get(int(layer_idx))
        if layer is None:
            rec = {"layer_idx": int(layer_idx), "status": "skipped", "reason": "layer_not_patched"}
            summary["skipped"].append(rec)
            if cfg.save_metrics:
                jsonl_append(metrics_path, rec)
            continue
        if layer_idx not in dense_targets:
            rec = {"layer_idx": int(layer_idx), "status": "skipped", "reason": "missing_dense_targets"}
            summary["skipped"].append(rec)
            if cfg.save_metrics:
                jsonl_append(metrics_path, rec)
            continue
        layer_result = fit_patched_ffn_layer(layer, dense_targets[layer_idx], cfg, logger)
        status = "success" if not layer_result["failed"] else ("partial_success" if layer_result["success"] else "failed")
        record = {"layer_idx": int(layer_idx), "status": status, **to_jsonable(layer_result)}
        summary["layers"].append(record)
        if layer_result["success"]:
            summary["successes"].append(int(layer_idx))
        if layer_result["failed"]:
            summary["failures"].append({"layer_idx": int(layer_idx), "projections": list(layer_result["failed"])})
        if cfg.save_metrics:
            jsonl_append(metrics_path, record)
    summary["finished_at"] = time.time()
    summary["elapsed_sec"] = max(0.0, float(summary["finished_at"] - summary["started_at"]))
    json_dump(summary_path, summary)
    return summary
