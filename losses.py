# Loss functions (PyTorch version)

import torch as tc
import torch.nn as nn
from typing import Optional

device = tc.device("cuda") if tc.cuda.is_available() else tc.device("cpu")


def log_softmax(x: tc.Tensor, temperature: float, dim: int = -1) -> tc.Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    x_fp32 = (x / tc.tensor(float(temperature), device=x.device, dtype=tc.float32)).to(tc.float32)
    return nn.functional.log_softmax(x_fp32, dim=dim).to(x.dtype)


def cross_entropy(logits: tc.Tensor, targets: tc.Tensor, temperature: float) -> tc.Tensor:
    logits = logits / temperature
    return nn.functional.cross_entropy(logits.to(device=device, dtype=tc.float32), targets.to(device=device).to(tc.long))


def mse_loss(pred: tc.Tensor, target: tc.Tensor) -> tc.Tensor:
    return nn.functional.mse_loss(pred.to(device=device, dtype=tc.float32), target.to(device=device, dtype=tc.float32))


def balance_loss(probs: tc.Tensor, mask: Optional[tc.Tensor], coeff: float) -> tc.Tensor:
    probs = probs.to(device=device, dtype=tc.float32)
    if mask is not None:
        mask = mask.to(device=device, dtype=tc.float32)
        masked = probs * mask
        denom = tc.sum(masked, dim=-1, keepdim=True) + 1e-9
        probs = masked / denom
    probs = tc.clamp(probs, 1e-9, 1.0)
    ent = -tc.sum(probs * tc.log(probs), dim=-1).mean()
    max_ent = tc.log(tc.tensor(probs.shape[-1], dtype=tc.float32, device=device))
    return coeff * (max_ent - ent)


def accuracy(logits: tc.Tensor, targets: tc.Tensor) -> float:
    preds = tc.argmax(logits, dim=-1)
    correct = (preds == targets).to(tc.float32)
    return float(correct.mean().detach().cpu().item())


def router_z_loss(logits: tc.Tensor, mask: tc.Tensor, coeff: float) -> tc.Tensor:
    logits = logits.to(device)
    if mask is None:
        return tc.zeros((), device=device, dtype=logits.dtype)
    mask = mask.to(device=device, dtype=logits.dtype)
    masked_logits = logits * mask
    z = tc.logsumexp(masked_logits, dim=-1)
    return coeff * (z ** 2).mean()
