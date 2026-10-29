"""Small-block variational quantum features for classical MoTE routing.

One qubit/readout per routed expert; disjoint blocks of at most four qubits.
Every gate is a real unitary (RY or controlled-RY). Exact local simulation
costs O(tokens * depth * experts * 2**block_size), never O(2**experts).
This is a quantum-circuit parameterization, not a quantum speed-up claim.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def apply_ry(state: torch.Tensor, angle: torch.Tensor, wire: int, width: int) -> torch.Tensor:
    """Apply RY to a batched state, using big-endian computational basis."""
    shape = state.shape
    split = state.reshape(*shape[:-1], 2**wire, 2, 2 ** (width - wire - 1))
    zero, one = split.unbind(-2)
    cosine = torch.cos(angle / 2)[..., None, None]
    sine = torch.sin(angle / 2)[..., None, None]
    return torch.stack((cosine * zero - sine * one, sine * zero + cosine * one), -2).reshape(shape)


def apply_cry(
    state: torch.Tensor, angle: torch.Tensor, control: int, target: int, width: int
) -> torch.Tensor:
    """Apply |0><0| I + |1><1| RY, with no dense unitary construction."""
    if control == target:
        raise ValueError("controlled-RY requires different control and target wires")
    basis = torch.arange(2**width, device=state.device)
    active = ((basis >> (width - control - 1)) & 1).bool()
    return torch.where(active, apply_ry(state, angle, target, width), state)


class QuantumGate(nn.Module):
    """Produce logits; TopKGate/SoftGate own normalization, masks and losses.

    The global state is a tensor product of small entangled blocks. Each layer
    re-uploads a shared input angle through trainable per-qubit scale/offset,
    followed by a nearest-neighbour CRY chain (direction alternates per layer).
    A final trainable RY sets the local measurement axis. Odd expert counts use
    a smaller final block: there are exactly E qubits, no dummy experts.

    ``born`` returns log(smoothed P(bit=1)); the existing temperature-softmax
    yields normalized marginal weights, not a joint Born categorical sample.
    ``expectation`` returns -<Z>, retaining an ordinary softmax ablation.
    """

    def __init__(
        self,
        data_dim: int,
        num_experts: int,
        *,
        rank: int = 16,
        depth: int = 2,
        block_size: int = 2,
        readout: str = "born",
        angle_scale: float = math.pi,
        probability_floor: float = 1e-4,
        norm: str = "none",
        output_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if min(int(data_dim), int(num_experts), int(rank)) < 1:
            raise ValueError("quantum gate dimensions and rank must be positive")
        if int(depth) not in range(1, 5):
            raise ValueError("gate_quantum_depth must be in [1, 4]")
        if int(block_size) not in {1, 2, 4}:
            raise ValueError("gate_quantum_block_size must be 1, 2 or 4")
        if readout not in {"born", "expectation"}:
            raise ValueError("gate_quantum_readout must be 'born' or 'expectation'")
        if not math.isfinite(angle_scale) or angle_scale <= 0:
            raise ValueError("gate_quantum_angle_scale must be finite and positive")
        if not math.isfinite(probability_floor) or not 0 < probability_floor < 0.5:
            raise ValueError("gate_quantum_probability_floor must be in (0, 0.5)")
        if norm not in {"none", "layernorm"}:
            raise ValueError("quantum gate norm must be 'none' or 'layernorm'")
        if not math.isfinite(output_init_std) or output_init_std <= 0:
            raise ValueError("quantum gate output_init_std must be finite and positive")
        self.data_dim, self.num_experts = int(data_dim), int(num_experts)
        self.rank, self.depth, self.block_size = int(rank), int(depth), int(block_size)
        self.readout, self.angle_scale = readout, float(angle_scale)
        self.probability_floor = float(probability_floor)
        self.norm = nn.LayerNorm(self.data_dim) if norm == "layernorm" else None
        self.input_proj = nn.Linear(self.data_dim, self.rank, bias=False)
        self.angle_proj = nn.Linear(self.rank, self.num_experts, bias=False)
        nn.init.normal_(self.angle_proj.weight, std=output_init_std)
        self.encoding_scale = nn.Parameter(torch.ones(self.depth, self.num_experts))
        self.rotation = nn.Parameter(torch.empty(self.depth, self.num_experts))
        nn.init.uniform_(self.rotation, -0.1, 0.1)
        groups = math.ceil(self.num_experts / self.block_size)
        self.coupling = nn.Parameter(torch.full((self.depth, self.num_experts - groups), 0.1))
        # Equatorial readout avoids the zero derivative at RY(0)|0> under Z.
        self.readout_rotation = nn.Parameter(torch.full((self.num_experts,), math.pi / 2))

    def _blocks(self, angles: torch.Tensor, start: int, groups: int, width: int, edge_start: int) -> torch.Tensor:
        end = start + groups * width
        data = angles[:, start:end].reshape(angles.shape[0], groups, width)
        dtype = angles.dtype
        scales = self.encoding_scale[:, start:end].to(dtype).reshape(self.depth, groups, width)
        offsets = self.rotation[:, start:end].to(dtype).reshape(self.depth, groups, width)
        coupling = self.coupling[:, edge_start:edge_start + groups * (width - 1)].to(dtype)
        coupling = coupling.reshape(self.depth, groups, width - 1)
        state = angles.new_zeros((angles.shape[0], groups, 2**width))
        state[..., 0] = 1
        for layer in range(self.depth):
            theta = data * scales[layer] + offsets[layer]
            for wire in range(width):
                state = apply_ry(state, theta[..., wire], wire, width)
            for edge in range(width - 1):
                control, target = (edge, edge + 1) if layer % 2 == 0 else (edge + 1, edge)
                state = apply_cry(state, coupling[layer, :, edge], control, target, width)
        axis = self.readout_rotation[start:end].to(dtype).reshape(groups, width)
        for wire in range(width):
            state = apply_ry(state, axis[:, wire], wire, width)
        basis = torch.arange(2**width, device=angles.device)
        shifts = torch.arange(width - 1, -1, -1, device=angles.device)
        signs = 1 - 2 * ((basis[:, None] >> shifts) & 1)
        return (state.square() @ signs.to(dtype)).reshape(angles.shape[0], groups * width)

    def expectations_from_angles(self, angles: torch.Tensor) -> torch.Tensor:
        """Exact local Z expectations; exposed for independent circuit checks."""
        if angles.shape[-1] != self.num_experts:
            raise ValueError(f"expected {self.num_experts} angles, got {angles.shape[-1]}")
        shape = angles.shape
        flat = angles.reshape(-1, self.num_experts)
        full, tail = divmod(self.num_experts, self.block_size)
        outputs = []
        if full:
            outputs.append(self._blocks(flat, 0, full, self.block_size, 0))
        if tail:
            outputs.append(self._blocks(flat, full * self.block_size, 1, tail, full * (self.block_size - 1)))
        return torch.cat(outputs, -1).reshape(shape)

    def expectations(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.data_dim:
            raise ValueError(f"expected input dimension {self.data_dim}, got {x.shape[-1]}")
        # Keep exact circuit arithmetic in fp32 even after an exported model is
        # cast to bf16/fp16. Preserve fp64 for numerical verification.
        dtype = torch.float64 if self.input_proj.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = x.to(dtype)
            if self.norm is not None:
                value = F.layer_norm(value, (self.data_dim,), self.norm.weight.to(dtype), self.norm.bias.to(dtype), self.norm.eps)
            code = F.linear(value, self.input_proj.weight.to(dtype))
            angles = self.angle_scale * torch.tanh(F.linear(code, self.angle_proj.weight.to(dtype)))
            return self.expectations_from_angles(angles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.expectations(x)
        if self.readout == "expectation":
            return -z
        # Smooth the marginal away from 0 and 1 before log; no sample noise.
        marginal = ((1 - z) / 2).clamp(0, 1)
        marginal = self.probability_floor + (1 - 2 * self.probability_floor) * marginal
        return marginal.log()
