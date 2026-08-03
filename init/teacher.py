from __future__ import annotations

import copy
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


MetricDict = Dict[str, float]
ProgressFn = Callable[[Dict[str, object]], None]


@torch.no_grad()
def evaluate_teacher_ffn(
    module: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    batch_tokens: int = 64,
) -> MetricDict:
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError(f"teacher tensors must be rank-2, got {inputs.shape=} {targets.shape=}")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("teacher input/target token counts differ")
    was_training = module.training
    module.eval()
    error_sq = 0.0
    target_sq = 0.0
    cosine_sum = 0.0
    value_count = 0
    token_count = 0
    for start in range(0, inputs.shape[0], max(1, int(batch_tokens))):
        stop = min(inputs.shape[0], start + max(1, int(batch_tokens)))
        x = inputs[start:stop].to(device=device, non_blocking=True)
        target = targets[start:stop].to(device=device, non_blocking=True)
        pred = module(x)
        pred_f = pred.float()
        target_f = target.float()
        error = pred_f - target_f
        error_sq += float(error.square().sum().item())
        target_sq += float(target_f.square().sum().item())
        value_count += int(error.numel())
        cosine = nn.functional.cosine_similarity(pred_f, target_f, dim=-1, eps=1e-8)
        cosine_sum += float(cosine.sum().item())
        token_count += int(cosine.numel())
    if was_training:
        module.train()
    mse = error_sq / max(1, value_count)
    target_rms = math.sqrt(target_sq / max(1, value_count))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "target_rms": target_rms,
        "nrmse": math.sqrt(mse) / max(1e-12, target_rms),
        "rel_l2": math.sqrt(error_sq) / max(1e-12, math.sqrt(target_sq)),
        "cosine": cosine_sum / max(1, token_count),
        "tokens": float(inputs.shape[0]),
    }


def fit_teacher_ffn(
    module: nn.Module,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    device: torch.device,
    steps: int = 300,
    batch_tokens: int = 32,
    eval_batch_tokens: int = 64,
    eval_every: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    progress: Optional[ProgressFn] = None,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    if train_inputs.shape[0] != train_targets.shape[0] or val_inputs.shape[0] != val_targets.shape[0]:
        raise ValueError("teacher input/target token counts differ")
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.AdamW(module.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    module.train()
    started = time.perf_counter()
    initial = evaluate_teacher_ffn(
        module,
        val_inputs,
        val_targets,
        device=device,
        batch_tokens=eval_batch_tokens,
    )
    best_metrics = dict(initial)
    best_step = 0
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}
    history: List[Dict[str, object]] = [{"step": 0, "val": dict(initial)}]
    last_loss = None
    last_grad_norm = None

    for step in range(1, int(steps) + 1):
        indices = torch.randint(
            0,
            int(train_inputs.shape[0]),
            (max(1, int(batch_tokens)),),
            generator=generator,
        )
        x = train_inputs.index_select(0, indices).to(device=device, non_blocking=True)
        target = train_targets.index_select(0, indices).to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = module(x)
        loss = (pred.float() - target.float()).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite teacher loss at step={step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), float(max_grad_norm))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite teacher grad norm at step={step}: {grad_norm}")
        optimizer.step()
        last_loss = float(loss.detach().item())
        last_grad_norm = float(grad_norm.detach().item())

        should_eval = step == int(steps) or step % max(1, int(eval_every)) == 0
        if should_eval:
            val = evaluate_teacher_ffn(
                module,
                val_inputs,
                val_targets,
                device=device,
                batch_tokens=eval_batch_tokens,
            )
            record: Dict[str, object] = {
                "step": step,
                "train_loss": last_loss,
                "grad_norm": last_grad_norm,
                "val": dict(val),
            }
            history.append(record)
            if float(val["rel_l2"]) < float(best_metrics["rel_l2"]):
                best_metrics = dict(val)
                best_step = step
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}
            if progress is not None:
                progress(record)

    final = evaluate_teacher_ffn(
        module,
        val_inputs,
        val_targets,
        device=device,
        batch_tokens=eval_batch_tokens,
    )
    module.load_state_dict(best_state, strict=True)
    result: Dict[str, object] = {
        "initial": initial,
        "final": final,
        "best": best_metrics,
        "best_step": int(best_step),
        "steps": int(steps),
        "batch_tokens": int(batch_tokens),
        "eval_batch_tokens": int(eval_batch_tokens),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "seed": int(seed),
        "last_train_loss": last_loss,
        "last_grad_norm": last_grad_norm,
        "elapsed_sec": time.perf_counter() - started,
        "history": history,
    }
    return result, best_state
