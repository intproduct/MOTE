"""Small structured routers; share routing/auxiliary loss logic in gate.py."""

from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .quantum_gate import apply_ry, apply_cry


def normalize_input(norm, x, dtype):
    x = x.to(dtype)
    if isinstance(norm, nn.LayerNorm):
        return F.layer_norm(
            x,
            norm.normalized_shape,
            norm.weight.to(dtype),
            norm.bias.to(dtype),
            norm.eps,
        )
    return x


class ResidualMPSGate(nn.Module):
    """Classical normalized matrix-product features, NOT a unitary circuit.

    M_i(x_i)=I+tanh(x_i) A_i/sqrt(m); the boundary vector contracts left to
    right and an output matrix returns E logits. Intermediate normalization
    preserves the final normalized direction (except the numerical epsilon).
    Sites m and bond dimension chi are independent of expert count.
    """

    def __init__(
        self,
        data_dim,
        num_experts,
        sites=16,
        bond_dim=8,
        norm="none",
        output_init_std=1e-3,
    ):
        super().__init__()
        if min(data_dim, num_experts, sites, bond_dim) < 1:
            raise ValueError("dimensions must be positive")
        self.sites, self.bond_dim = sites, bond_dim
        self.norm = nn.LayerNorm(data_dim) if norm == "layernorm" else nn.Identity()
        self.input_proj = nn.Linear(data_dim, sites, bias=False)
        self.cores = nn.Parameter(
            torch.randn(sites, bond_dim, bond_dim) / math.sqrt(bond_dim)
        )
        self.boundary = nn.Parameter(F.normalize(torch.randn(bond_dim), dim=0))
        self.output_proj = nn.Linear(bond_dim, num_experts, bias=False)
        nn.init.normal_(self.output_proj.weight, std=output_init_std)

    def forward(self, x):
        dtype = (
            torch.float64
            if self.input_proj.weight.dtype == torch.float64
            else torch.float32
        )
        with torch.autocast(x.device.type, enabled=False):
            values = torch.tanh(
                F.linear(
                    normalize_input(self.norm, x, dtype),
                    self.input_proj.weight.to(dtype),
                )
            )
            state = self.boundary.to(dtype).expand(*values.shape[:-1], self.bond_dim)
            cores = self.cores.to(dtype)
            for i in range(self.sites):
                state = state + values[..., i, None] * (state @ cores[i]) / math.sqrt(
                    self.sites
                )
                state = F.normalize(state, dim=-1, eps=1e-8)
            return F.linear(state, self.output_proj.weight.to(dtype))


def apply_rz(state, angle, wire, width):
    shape = state.shape
    split = state.reshape(*shape[:-1], 2**wire, 2, 2 ** (width - wire - 1))
    zero, one = split.unbind(-2)
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    phase0 = torch.complex(c, -s)[..., None, None]
    phase1 = torch.complex(c, s)[..., None, None]
    return torch.stack((phase0 * zero, phase1 * one), -2).reshape(shape)


class CompactBornGate(nn.Module):
    """log2(E) qubits, one joint computational-basis outcome per expert.

    Exact statevector contraction: E amplitudes for E experts (E=2..64).
    RY/RZ data re-uploading, alternating nearest-neighbor bidirectional CRY,
    final RY, then joint Z-basis measurement. No sampling or speedup claim.
    """

    def __init__(
        self,
        data_dim,
        num_experts,
        rank=16,
        depth=2,
        prob_floor=1e-6,
        max_qubits=6,
        norm="none",
        output_init_std=1e-3,
        entangle=True,
    ):
        super().__init__()
        if min(data_dim, rank) < 1 or num_experts < 2 or depth not in {1, 2, 3, 4}:
            raise ValueError("positive dimensions, E>=2 and depth in [1,4] required")
        if num_experts & (num_experts - 1) or num_experts > 2**max_qubits:
            raise ValueError(
                "compact_born requires a power-of-two expert count from 2 to 2**max_qubits (default 64)"
            )
        if not 0 < prob_floor < 1 / num_experts:
            raise ValueError("prob_floor must be positive and smaller than 1/E")
        self.entangle = bool(entangle)
        self.rank, self.prob_floor = rank, prob_floor
        self.norm = nn.LayerNorm(data_dim) if norm == "layernorm" else nn.Identity()
        self.num_experts, self.depth = num_experts, depth
        self.qubits = (num_experts - 1).bit_length()
        self.input_proj = nn.Linear(data_dim, rank, bias=False)
        self.angle_proj = nn.Linear(rank, 2 * depth * self.qubits, bias=False)
        nn.init.normal_(self.angle_proj.weight, std=output_init_std)
        self.rotation = nn.Parameter(
            torch.empty(depth, self.qubits, 2).uniform_(-0.05, 0.05)
        )
        with torch.no_grad():
            self.rotation[0, :, 0].add_(math.pi / 4)
        edges = sum(
            2 * len(range(layer % 2, self.qubits - 1, 2)) for layer in range(depth)
        )
        self.coupling = nn.Parameter(torch.full((edges if self.entangle else 0,), 0.05))
        self.readout_rotation = nn.Parameter(torch.full((self.qubits,), math.pi / 4))

    def probabilities_from_angles(self, angles):
        """Return normalized joint probabilities in big-endian bitstring order."""
        leading = angles.shape[:-3]
        a = angles.reshape(-1, self.depth, self.qubits, 2)
        complex_dtype = (
            torch.complex128 if angles.dtype == torch.float64 else torch.complex64
        )
        state = torch.zeros(
            (a.shape[0], 2**self.qubits), device=a.device, dtype=complex_dtype
        )
        state[:, 0] = 1
        theta = a + self.rotation.to(angles.dtype)
        gamma = self.coupling.to(angles.dtype)
        edge = 0
        for layer in range(self.depth):
            for q in range(self.qubits):
                state = apply_ry(state, theta[:, layer, q, 0], q, self.qubits)
                state = apply_rz(state, theta[:, layer, q, 1], q, self.qubits)
            for q in range(layer % 2, self.qubits - 1, 2) if self.entangle else ():
                c, t = (q, q + 1) if layer % 2 == 0 else (q + 1, q)
                state = apply_cry(state, gamma[edge], c, t, self.qubits)
                state = apply_cry(state, gamma[edge + 1], t, c, self.qubits)
                edge += 2
        for q in range(self.qubits):
            state = apply_ry(
                state, self.readout_rotation[q].to(angles.dtype), q, self.qubits
            )
        return (state.real.square() + state.imag.square()).reshape(
            *leading, 2**self.qubits
        )

    def forward(self, x):
        dtype = (
            torch.float64
            if self.input_proj.weight.dtype == torch.float64
            else torch.float32
        )
        with torch.autocast(x.device.type, enabled=False):
            code = F.linear(
                normalize_input(self.norm, x, dtype), self.input_proj.weight.to(dtype)
            )
            angles = math.pi * torch.tanh(
                F.linear(code, self.angle_proj.weight.to(dtype))
            )
            angles = angles.reshape(*x.shape[:-1], self.depth, self.qubits, 2)
            full = self.probabilities_from_angles(angles)
            # Normalized smoothing: at gain=temperature=1, softmax(logits)=p.
            probs = (full + self.prob_floor) / (1 + self.num_experts * self.prob_floor)
            logits = probs.log()
            return logits
