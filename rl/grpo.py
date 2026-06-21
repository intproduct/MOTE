from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def compute_group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """Normalize rewards inside each prompt group, never across the whole batch.

    Args:
        rewards: Reward tensor shaped [batch_size, group_size], or a flat tensor
            whose length is divisible by group_size.
        group_size: Number of sampled responses per prompt.
        eps: Minimum std for a group to be considered non-degenerate.

    Returns:
        Advantage tensor with the same shape as rewards. Degenerate groups with
        near-zero std get zero advantages to avoid NaNs and meaningless updates.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    orig_shape = rewards.shape
    if rewards.ndim == 1:
        if rewards.numel() % int(group_size) != 0:
            raise ValueError(f"flat rewards length {rewards.numel()} is not divisible by group_size={group_size}")
        grouped = rewards.reshape(-1, int(group_size))
    elif rewards.ndim == 2:
        if int(rewards.shape[1]) != int(group_size):
            raise ValueError(f"rewards second dim must equal group_size={group_size}, got shape={tuple(rewards.shape)}")
        grouped = rewards
    else:
        raise ValueError(f"rewards must be 1D or 2D, got shape={tuple(rewards.shape)}")

    grouped = grouped.to(dtype=torch.float32)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    safe = std > float(eps)
    advantages = torch.where(safe, (grouped - mean) / std.clamp_min(float(eps)), torch.zeros_like(grouped))
    return advantages.reshape(orig_shape)


def _per_response_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average tokens inside each response first, then average responses.

    GRPO treats each sampled response as one policy sample. A direct global
    token masked mean would overweight longer responses, so every [N, T] tensor
    is reduced to [N] by response-token mean before the final response mean.
    """
    mask = mask.to(dtype=values.dtype, device=values.device)
    denom = mask.sum(dim=1).clamp_min(1.0)
    per_response = (values * mask).sum(dim=1) / denom
    return per_response.mean()


def _expand_advantages(advantages: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    adv = advantages.to(dtype=target.dtype, device=target.device)
    if adv.ndim == 2:
        adv = adv.reshape(-1)
    if adv.ndim == 1:
        if int(adv.numel()) != int(target.shape[0]):
            raise ValueError(f"advantages length {adv.numel()} must match response count {target.shape[0]}")
        adv = adv.unsqueeze(1)
    if adv.shape == (target.shape[0], 1):
        return adv.expand_as(target)
    if adv.shape == target.shape:
        return adv
    raise ValueError(
        "advantages must have shape [B, G], [B*G], [B*G, 1], or [B*G, T]; "
        f"got {tuple(advantages.shape)} for target {tuple(target.shape)}"
    )


def grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: Optional[torch.Tensor],
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    eps_clip: float = 0.2,
    beta: float = 0.04,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """PPO-style clipped GRPO token loss with optional non-negative KL estimate."""
    if new_logprobs.shape != old_logprobs.shape:
        raise ValueError(f"new/old logprob shapes differ: {tuple(new_logprobs.shape)} vs {tuple(old_logprobs.shape)}")
    if response_mask.shape != new_logprobs.shape:
        raise ValueError(f"response_mask shape {tuple(response_mask.shape)} must match logprobs {tuple(new_logprobs.shape)}")
    if float(beta) != 0.0 and ref_logprobs is None:
        raise ValueError("ref_logprobs is required when beta is non-zero")
    if ref_logprobs is not None and ref_logprobs.shape != new_logprobs.shape:
        raise ValueError(f"ref_logprobs shape {tuple(ref_logprobs.shape)} must match logprobs {tuple(new_logprobs.shape)}")

    old_logprobs = old_logprobs.detach()
    mask = response_mask.to(dtype=new_logprobs.dtype, device=new_logprobs.device)
    adv = _expand_advantages(advantages, new_logprobs)

    raw_log_ratio = new_logprobs - old_logprobs
    raw_log_ratio_abs = raw_log_ratio.detach().abs()
    raw_log_ratio_max = torch.where(mask.bool(), raw_log_ratio_abs, torch.zeros_like(raw_log_ratio_abs)).max()
    log_ratio = raw_log_ratio.clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(min=1.0 - float(eps_clip), max=1.0 + float(eps_clip))

    surrogate = torch.minimum(ratio * adv, clipped_ratio * adv)
    policy_loss = -_per_response_masked_mean(surrogate, mask)

    if float(beta) != 0.0 and ref_logprobs is not None:
        ref_delta = (ref_logprobs.detach() - new_logprobs).clamp(min=-20.0, max=20.0)
        kl = torch.exp(ref_delta) - ref_delta - 1.0
        kl_mean = _per_response_masked_mean(kl, mask)
        loss = policy_loss + float(beta) * kl_mean
    else:
        kl_mean = new_logprobs.new_zeros(())
        loss = policy_loss

    clipped = (ratio - 1.0).abs() > float(eps_clip)
    metrics = {
        "policy_loss": policy_loss.detach(),
        "kl_mean": kl_mean.detach(),
        "clip_frac": _per_response_masked_mean(clipped.to(dtype=new_logprobs.dtype), mask).detach(),
        "ratio_mean": _per_response_masked_mean(ratio, mask).detach(),
        "ratio_max": torch.where(mask.bool(), ratio.detach(), torch.zeros_like(ratio.detach())).max(),
        "raw_log_ratio_max": raw_log_ratio_max.detach(),
        "adv_abs_mean": _per_response_masked_mean(adv.abs(), mask).detach(),
    }
    return loss, metrics
