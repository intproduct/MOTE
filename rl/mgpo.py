from __future__ import annotations

import torch


def compute_mgpo_weights(
    prompt_acc: torch.Tensor,
    p0: float = 0.5,
    gamma: float = 2.0,
    weight_min: float = 0.1,
    weight_max: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute MGPO prompt weights from Bernoulli accuracy estimates."""
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if float(weight_min) > float(weight_max):
        raise ValueError(f"weight_min must be <= weight_max, got {weight_min} > {weight_max}")

    p = prompt_acc.to(dtype=torch.float32).clamp(float(eps), 1.0 - float(eps))
    p0_tensor = torch.as_tensor(float(p0), dtype=p.dtype, device=p.device).clamp(float(eps), 1.0 - float(eps))
    one = torch.ones_like(p)
    d_me = p * torch.log(p / p0_tensor) + (one - p) * torch.log((one - p) / (one - p0_tensor))
    weights = torch.exp(-float(gamma) * d_me)
    return weights.clamp(float(weight_min), float(weight_max))
