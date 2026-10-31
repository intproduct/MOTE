"""Four FP32 routers with identical hard-top-k dispatch semantics.

Basis order is big endian and expert assignment is the identity. This module
reuses the frozen RY primitive, never a historically trained router or assignment.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from ...quantum_gate import apply_ry

METHODS = ("mlp", "affine_born", "reencode_cz", "reencode_product")


def reencoded_state(angles: torch.Tensor, entangle: bool) -> torch.Tensor:
    """Pure differentiable state function; also supports float64 references."""
    if angles.shape[-1] != 8:
        raise ValueError("Four wires require eight re-encoding angles")
    angles = angles.reshape(-1, 8)
    state = angles.new_zeros(len(angles), 16)
    state[:, 0] = 1
    for wire in range(4):
        state = apply_ry(state, angles[:, wire], wire, 4)
    if entangle:
        basis = torch.arange(16, device=angles.device)
        bits = (basis[:, None] >> torch.arange(3, -1, -1, device=angles.device)) & 1
        state = state * (1 - 2 * bits[:, :-1] * bits[:, 1:]).prod(1)
    for wire in range(4):
        state = apply_ry(state, angles[:, wire + 4], wire, 4)
    return state


@dataclass
class Routing:
    indices: torch.Tensor
    weights: torch.Tensor
    probabilities: torch.Tensor
    load_balance: torch.Tensor


def hard_topk(logits: torch.Tensor, k: int) -> Routing:
    """Stable index tie break; gradients only through selected task logits.

    There is no STE, token dropping, capacity limit, noise or temperature change.
    The separately reported load-balancing term uses all softmax probabilities.
    """
    if logits.ndim != 2 or not 0 < k <= logits.shape[-1]:
        raise ValueError("Expected [tokens, experts] logits and a valid k")
    probabilities = logits.softmax(-1)
    indices = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :k]
    weights = logits.gather(1, indices).softmax(-1)
    frequency = torch.nn.functional.one_hot(indices, logits.shape[-1]).float().mean((0, 1))
    balance = logits.shape[-1] * (probabilities.mean(0) * frequency).sum()
    return Routing(indices, weights, probabilities, balance)


class JointRouter(nn.Module):
    def __init__(self, input_dim: int, method: str, seed: int, rank: int = 8, top_k: int = 4):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method, self.top_k = method, top_k
        # Construction consumes no caller RNG; compatible encoders/heads are paired.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.norm = nn.LayerNorm(input_dim)
            self.input_proj = nn.Linear(input_dim, rank, bias=False)
            self.output = nn.Linear(rank, 8 if method.startswith("reencode") else 16)
            nn.init.normal_(self.output.weight, std=0.01)
            nn.init.zeros_(self.output.bias)
        self.log_scale = nn.Parameter(torch.zeros(()))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise TypeError("Router parameters must remain FP32")
        with torch.autocast(device_type=x.device.type, enabled=False):
            v = self.output(torch.tanh(self.input_proj(self.norm(x.float()))))
            if self.method == "mlp":
                logits = v
            else:
                if self.method == "affine_born":
                    p = (1 + v).square()
                else:
                    offset = v.new_tensor([math.pi / 2] * 4 + [0.35] * 4)
                    p = reencoded_state(offset + (math.pi / 2) * v.tanh(),
                                        self.method == "reencode_cz").square()
                p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
                p = (p + 1e-6) / (1 + 16e-6)
                logits = p.log()  # Negative of the frozen research's centered cost.
            return (logits - logits.mean(-1, keepdim=True)) * self.log_scale.clamp(-3, 3).exp()

    def forward(self, x: torch.Tensor) -> Routing:
        return hard_topk(self.logits(x.reshape(-1, x.shape[-1])), self.top_k)
