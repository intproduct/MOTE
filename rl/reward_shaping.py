from __future__ import annotations

import torch


def apply_long2short_reward_shift(
    reward_tensor: torch.Tensor,
    response_lens: torch.Tensor,
    lambda_value: float = 0.2,
    min_correct: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Redistribute reward among correct responses to prefer shorter traces."""
    if reward_tensor.ndim != 2:
        raise ValueError(f"reward_tensor must have shape [batch_size, group_size], got {tuple(reward_tensor.shape)}")
    if int(min_correct) < 1:
        raise ValueError(f"min_correct must be >= 1, got {min_correct}")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")

    rewards = reward_tensor.to(dtype=torch.float32)
    lens = response_lens.to(dtype=torch.float32, device=rewards.device)
    if lens.ndim == 1:
        if int(lens.numel()) != int(rewards.numel()):
            raise ValueError(f"flat response_lens length {lens.numel()} must match reward count {rewards.numel()}")
        lens = lens.reshape_as(rewards)
    elif lens.shape != rewards.shape:
        raise ValueError(f"response_lens shape {tuple(lens.shape)} must match reward_tensor {tuple(rewards.shape)}")

    shaped = rewards.clone()
    safe_lens = lens.clamp_min(float(eps))
    for batch_idx in range(rewards.shape[0]):
        correct = rewards[batch_idx] == 1.0
        if int(correct.sum().item()) < int(min_correct):
            continue
        scores = 1.0 / safe_lens[batch_idx, correct]
        centered = scores - scores.mean()
        denom = centered.abs().max()
        if not torch.isfinite(denom) or float(denom.item()) <= float(eps):
            continue
        shaped[batch_idx, correct] = rewards[batch_idx, correct] + float(lambda_value) * centered / denom
    return shaped
