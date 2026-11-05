"""Supervised and distillation objectives shared independently of the Trainer."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .loss_metrics import DEFERRED_LOSS_METRICS_MARKER, materialize_loss_observability

KNOWN_BUCKETS = ("pretrain_general", "gsm8k_core", "aux_reasoning", "unknown_bucket")


@torch.no_grad()
def build_teacher_topk_targets(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    topk: int,
    token_chunk_size: int,
) -> dict:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(f"Expected logits [B,L,V] and labels [B,L], got {logits.shape}, {labels.shape}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"Teacher logits/labels shape mismatch: {logits.shape[:2]} != {labels.shape}")
    vocab_size = int(logits.shape[-1])
    resolved_topk = min(int(topk), vocab_size)
    if resolved_topk < 1:
        raise ValueError(f"Invalid KD top-k={topk} for vocab_size={vocab_size}")

    shift_logits = logits[..., :-1, :]
    valid = labels[..., 1:].ne(-100)
    positions = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    if positions.numel() == 0:
        raise ValueError("KD microbatch has no supervised target tokens")
    flat_logits = shift_logits.reshape(-1, vocab_size)
    top_indices = []
    top_log_probs = []
    residual_log_probs = []
    eps = torch.finfo(torch.float32).eps
    for start in range(0, int(positions.numel()), int(token_chunk_size)):
        chunk_positions = positions[start : start + int(token_chunk_size)]
        scaled = flat_logits.index_select(0, chunk_positions).float() / float(temperature)
        values, indices = torch.topk(scaled, k=resolved_topk, dim=-1)
        log_z = torch.logsumexp(scaled, dim=-1, keepdim=True)
        log_probs = values - log_z
        top_mass = log_probs.exp().sum(dim=-1).clamp(min=0.0, max=1.0 - eps)
        top_indices.append(indices)
        top_log_probs.append(log_probs)
        residual_log_probs.append(torch.log1p(-top_mass))
    all_top_log_probs = torch.cat(top_log_probs, dim=0)
    all_residual_log_probs = torch.cat(residual_log_probs, dim=0)
    return {
        "top_indices": torch.cat(top_indices, dim=0),
        "top_log_probs": all_top_log_probs,
        "residual_log_probs": all_residual_log_probs,
        "valid_tokens": int(positions.numel()),
        "topk": resolved_topk,
        "mean_topk_mass": all_top_log_probs.exp().sum(dim=-1).mean(),
    }

def compute_exact_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    token_chunk_size: int,
    divergence: str = "forward_kl",
    jsd_beta: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    if student_logits.ndim != 3 or teacher_logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Exact KD expects student/teacher logits [B,L,V] and labels [B,L]")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"Exact KD teacher/student shape mismatch: {teacher_logits.shape} != {student_logits.shape}")
    if student_logits.shape[:2] != labels.shape:
        raise ValueError(f"Exact KD logits/labels shape mismatch: {student_logits.shape[:2]} != {labels.shape}")

    vocab_size = int(student_logits.shape[-1])
    positions = labels[..., 1:].ne(-100).reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    valid_tokens = int(positions.numel())
    if valid_tokens == 0:
        raise ValueError("Exact KD microbatch has no supervised target tokens")
    student_flat = student_logits[..., :-1, :].reshape(-1, vocab_size)
    teacher_flat = teacher_logits[..., :-1, :].reshape(-1, vocab_size)
    student_device = student_logits.device
    total_kl = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, valid_tokens, int(token_chunk_size)):
        chunk_positions = positions[start : start + int(token_chunk_size)]
        teacher_positions = chunk_positions.to(device=teacher_logits.device, non_blocking=True)
        teacher_scaled = teacher_flat.index_select(0, teacher_positions).float() / float(temperature)
        teacher_log_probs = F.log_softmax(teacher_scaled, dim=-1).to(
            device=student_device, non_blocking=True
        )
        student_scaled = student_flat.index_select(0, chunk_positions).float() / float(temperature)
        student_log_probs = F.log_softmax(student_scaled, dim=-1)
        resolved_divergence = str(divergence).strip().lower()
        if resolved_divergence == "forward_kl":
            chunk_loss = F.kl_div(student_log_probs, teacher_log_probs.exp(), reduction="sum")
        elif resolved_divergence == "reverse_kl":
            student_probs = student_log_probs.exp()
            chunk_loss = (student_probs * (student_log_probs - teacher_log_probs)).sum()
        elif resolved_divergence == "jsd":
            beta = float(jsd_beta)
            if not 0.0 < beta < 1.0:
                raise ValueError(f"jsd_beta must be in (0, 1), got {beta}")
            teacher_probs = teacher_log_probs.exp()
            student_probs = student_log_probs.exp()
            mixture_probs = beta * teacher_probs + (1.0 - beta) * student_probs
            mixture_log_probs = mixture_probs.clamp_min(torch.finfo(torch.float32).tiny).log()
            chunk_loss = beta * (teacher_probs * (teacher_log_probs - mixture_log_probs)).sum()
            chunk_loss = chunk_loss + (1.0 - beta) * (
                student_probs * (student_log_probs - mixture_log_probs)
            ).sum()
        else:
            raise ValueError(
                f"Unsupported exact KD divergence={divergence!r}; expected forward_kl, reverse_kl, or jsd"
            )
        total_kl = total_kl + chunk_loss
    loss = total_kl * (float(temperature) ** 2) / valid_tokens
    return loss, {
        "kd_valid_tokens": valid_tokens,
        "kd_distribution": "exact",
        "kd_divergence": str(divergence).strip().lower(),
        "kd_jsd_beta": float(jsd_beta) if str(divergence).strip().lower() == "jsd" else None,
    }

def compute_topk_distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_targets: dict,
    *,
    temperature: float,
    token_chunk_size: int,
) -> tuple[torch.Tensor, dict]:
    if student_logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"Expected student logits [B,L,V] and labels [B,L], got {student_logits.shape}, {labels.shape}"
        )
    if student_logits.shape[:2] != labels.shape:
        raise ValueError(f"Student logits/labels shape mismatch: {student_logits.shape[:2]} != {labels.shape}")
    vocab_size = int(student_logits.shape[-1])
    shift_logits = student_logits[..., :-1, :]
    positions = labels[..., 1:].ne(-100).reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    valid_tokens = int(positions.numel())
    if valid_tokens != int(teacher_targets["valid_tokens"]):
        raise ValueError(
            f"Teacher/student KD token count mismatch: {teacher_targets['valid_tokens']} != {valid_tokens}"
        )
    if valid_tokens == 0:
        raise ValueError("KD microbatch has no supervised target tokens")

    device = student_logits.device
    teacher_indices = teacher_targets["top_indices"].to(device=device, non_blocking=True)
    teacher_top_log_probs = teacher_targets["top_log_probs"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    teacher_residual_log_probs = teacher_targets["residual_log_probs"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    if teacher_indices.shape[0] != valid_tokens:
        raise ValueError("Teacher top-k rows do not match supervised token count")
    if int(teacher_indices.max().item()) >= vocab_size:
        raise ValueError("Teacher top-k token id exceeds student vocabulary")

    flat_logits = shift_logits.reshape(-1, vocab_size)
    total_kl = student_logits.new_zeros((), dtype=torch.float32)
    eps = torch.finfo(torch.float32).eps
    for start in range(0, valid_tokens, int(token_chunk_size)):
        end = min(start + int(token_chunk_size), valid_tokens)
        chunk_positions = positions[start:end]
        scaled = flat_logits.index_select(0, chunk_positions).float() / float(temperature)
        student_log_z = torch.logsumexp(scaled, dim=-1, keepdim=True)
        indices = teacher_indices[start:end]
        student_top_log_probs = scaled.gather(dim=-1, index=indices) - student_log_z
        student_top_mass = student_top_log_probs.exp().sum(dim=-1).clamp(
            min=0.0, max=1.0 - eps
        )
        student_residual_log_probs = torch.log1p(-student_top_mass)

        teacher_chunk_log_probs = teacher_top_log_probs[start:end]
        teacher_chunk_probs = teacher_chunk_log_probs.exp()
        teacher_chunk_residual_log_probs = teacher_residual_log_probs[start:end]
        teacher_chunk_residual_probs = teacher_chunk_residual_log_probs.exp()

        cross_entropy = -(
            (teacher_chunk_probs * student_top_log_probs).sum(dim=-1)
            + teacher_chunk_residual_probs * student_residual_log_probs
        )
        teacher_entropy = -(
            (teacher_chunk_probs * teacher_chunk_log_probs).sum(dim=-1)
            + teacher_chunk_residual_probs * teacher_chunk_residual_log_probs
        )
        total_kl = total_kl + (cross_entropy - teacher_entropy).sum()

    loss = total_kl * (float(temperature) ** 2) / valid_tokens
    return loss, {
        "kd_valid_tokens": valid_tokens,
        "kd_topk": int(teacher_targets["topk"]),
        "kd_mean_topk_mass": teacher_targets["mean_topk_mass"],
    }

def compute_loss_observability(
    logits,
    labels,
    *,
    buckets=None,
    tasks=None,
    loss_weights=None,
    collect_metrics=True,
    defer_metrics=False,
):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = int(shift_logits.shape[-1])
    token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    valid = shift_labels.ne(-100)
    valid_tokens = valid.sum().clamp_min(1)
    standard_loss = token_loss.sum() / valid_tokens

    final_mask = torch.zeros_like(valid)
    if loss_weights is not None:
        shift_weights = loss_weights[..., 1:].to(device=token_loss.device, dtype=token_loss.dtype).contiguous()
        shift_weights = torch.where(valid, shift_weights, torch.zeros_like(shift_weights))
        denom = shift_weights.sum()
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        weighted_loss = (token_loss * shift_weights).sum() / safe_denom
        loss = torch.where(denom > 0, weighted_loss, standard_loss)
        final_mask = valid & shift_weights.gt(1.0)
    else:
        loss = standard_loss

    if not collect_metrics:
        return loss, {}

    token_loss_fp32 = token_loss.float()
    zero_loss = token_loss_fp32.new_zeros(())
    zero_count = valid.new_zeros((), dtype=torch.long)
    bucket_loss_sums = {bucket: zero_loss for bucket in KNOWN_BUCKETS}
    bucket_token_counts = {bucket: zero_count for bucket in KNOWN_BUCKETS}
    task_loss_sums = {}
    task_token_counts = {}
    row_loss_sums = token_loss_fp32.sum(dim=-1).detach()
    row_token_counts = valid.sum(dim=-1).detach()
    if buckets is not None:
        bucket_names = list(buckets)
        for row_idx, bucket_name in enumerate(bucket_names[: token_loss.shape[0]]):
            key_bucket = str(bucket_name or "unknown_bucket")
            if key_bucket not in KNOWN_BUCKETS:
                key_bucket = "unknown_bucket"
            bucket_loss_sums[key_bucket] = bucket_loss_sums[key_bucket] + row_loss_sums[row_idx]
            bucket_token_counts[key_bucket] = bucket_token_counts[key_bucket] + row_token_counts[row_idx]
    if tasks is not None:
        for row_idx, task_name in enumerate(list(tasks)[: token_loss.shape[0]]):
            key_task = str(task_name or "unknown")
            task_loss_sums[key_task] = task_loss_sums.get(key_task, zero_loss) + row_loss_sums[row_idx]
            task_token_counts[key_task] = task_token_counts.get(key_task, zero_count) + row_token_counts[row_idx]
    deferred = {
        DEFERRED_LOSS_METRICS_MARKER: True,
        "bucket_loss_sums": bucket_loss_sums,
        "bucket_token_counts": bucket_token_counts,
        "task_loss_sums": task_loss_sums,
        "task_token_counts": task_token_counts,
        "final_answer_loss_sum": (token_loss_fp32 * final_mask).sum().detach(),
        "final_answer_weighted_tokens": final_mask.sum().detach(),
    }
    return loss, deferred if defer_metrics else materialize_loss_observability(deferred)
