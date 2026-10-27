"""Direct joint-state TN routers for parameter-efficiency experiments.

ResidualMPOGate is classical and nonlinear. AmplitudeReuploadBornGate uses
normalized real amplitudes and exact orthogonal RY/CRY evolution. Neither
implementation claims quantum hardware acceleration.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from .structured_routers import normalize_input
from .quantum_gate import apply_ry, apply_cry


def _check_dimensions(data_dim, num_experts, rank, depth):
    if min(data_dim, rank) < 1 or depth not in range(1, 5):
        raise ValueError('positive dimensions and depth=1..4 required')
    if num_experts < 2 or num_experts > 64 or num_experts & (num_experts - 1):
        raise ValueError('expert count must be a power of two in [2,64]')


class MPOOperator(nn.Module):
    """Open-boundary MPO: cores [chi_left, out_bit, in_bit, chi_right].

    Construct one shared E x E operator per forward, then use a GEMM over all
    tokens. This is exact, differentiable, and does not truncate the state.
    """
    def __init__(self, qubits, bond_dim):
        super().__init__()
        if qubits < 1 or bond_dim < 1:
            raise ValueError('qubits and bond_dim must be positive')
        self.qubits, self.bond_dim = qubits, bond_dim
        bonds = [1] + [bond_dim] * (qubits - 1) + [1]
        self.cores = nn.ParameterList([
            nn.Parameter(torch.randn(bonds[i], 2, 2, bonds[i+1]) /
                         math.sqrt(2 * bonds[i]))
            for i in range(qubits)
        ])

    def matrix(self, dtype=None):
        value = self.cores[0].to(dtype=dtype).squeeze(0)
        for core in self.cores[1:]:
            value = torch.tensordot(value, core.to(dtype=dtype), dims=([-1], [0]))
        value = value.squeeze(-1)
        order = list(range(0, 2*self.qubits, 2)) + list(range(1, 2*self.qubits, 2))
        return value.permute(order).reshape(2**self.qubits, 2**self.qubits)

    def forward(self, state):
        return F.linear(state, self.matrix(state.dtype))


class ResidualMPOGate(nn.Module):
    """D -> rank -> E joint coordinates, nonlinear MPO blocks, linear readout.

    a_0=V tanh(P LN(x)); a_{l+1}=a_l+lambda_l tanh(MPO_l a_l).
    The E-dimensional joint state is equivalent to q binary physical axes.
    A final E->E head produces logits; it is included in parameter budgets.
    """
    def __init__(self, data_dim, num_experts, rank=16, depth=2, bond_dim=2,
                 norm='none', output_init_std=1e-3):
        super().__init__()
        _check_dimensions(data_dim, num_experts, rank, depth)
        self.rank, self.depth, self.bond_dim = rank, depth, bond_dim
        self.num_experts = num_experts
        self.norm = nn.LayerNorm(data_dim) if norm == 'layernorm' else nn.Identity()
        self.input_proj = nn.Linear(data_dim, rank, bias=False)
        self.amplitude_proj = nn.Linear(rank, num_experts, bias=False)
        self.operators = nn.ModuleList([
            MPOOperator(num_experts.bit_length()-1, bond_dim) for _ in range(depth)
        ])
        self.residual_scale = nn.Parameter(torch.full((depth,), 0.1))
        self.readout = nn.Linear(num_experts, num_experts, bias=False)
        nn.init.normal_(self.readout.weight, std=output_init_std)

    def forward(self, x):
        dtype = torch.float64 if self.input_proj.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(x.device.type, enabled=False):
            code = torch.tanh(F.linear(normalize_input(self.norm, x, dtype),
                                      self.input_proj.weight.to(dtype)))
            state = F.linear(code, self.amplitude_proj.weight.to(dtype))
            for scale, operator in zip(self.residual_scale.to(dtype), self.operators):
                state = state + scale * torch.tanh(operator(state))
            return F.linear(state, self.readout.weight.to(dtype))


class AmplitudeReuploadBornGate(nn.Module):
    """Joint amplitude encoding followed by real unitary data re-uploading.

    Unlike CompactBornGate, the initial state already contains E learned
    amplitudes. All evolution after state preparation is RY/CRY (real unitary).
    A small positive offset avoids an undefined zero input state. Full Born
    readout is normalized and smoothed before returning log probabilities.
    """
    def __init__(self, data_dim, num_experts, rank=16, depth=2, entangle=True,
                 prob_floor=1e-6, norm='none', output_init_std=1e-3):
        super().__init__()
        _check_dimensions(data_dim, num_experts, rank, depth)
        if not 0 < prob_floor < 1/num_experts:
            raise ValueError('prob_floor must be in (0,1/E)')
        self.rank, self.depth, self.num_experts = rank, depth, num_experts
        self.qubits = num_experts.bit_length()-1
        self.entangle, self.prob_floor = bool(entangle), prob_floor
        self.norm = nn.LayerNorm(data_dim) if norm == 'layernorm' else nn.Identity()
        self.input_proj = nn.Linear(data_dim, rank, bias=False)
        self.amplitude_proj = nn.Linear(rank, num_experts, bias=False)
        self.angle_proj = nn.Linear(rank, depth*self.qubits, bias=False)
        nn.init.normal_(self.amplitude_proj.weight, std=output_init_std)
        nn.init.normal_(self.angle_proj.weight, std=output_init_std)
        self.rotation = nn.Parameter(torch.zeros(depth, self.qubits))
        edges = sum(2*len(range(layer % 2, self.qubits-1, 2)) for layer in range(depth))
        # Zero coupling preserves a near-uniform start without blocking its gradient.
        self.coupling = nn.Parameter(torch.zeros(edges if self.entangle else 0))

    def evolve(self, state, angles):
        """Preserve arbitrary leading batch axes; angles [..., depth, qubits]."""
        theta = angles + self.rotation.to(angles.dtype)
        edge = 0
        for layer in range(self.depth):
            for wire in range(self.qubits):
                state = apply_ry(state, theta[..., layer, wire], wire, self.qubits)
            for wire in range(layer % 2, self.qubits-1, 2) if self.entangle else ():
                control, target = (wire, wire+1) if layer % 2 == 0 else (wire+1, wire)
                state = apply_cry(state, self.coupling[edge].to(angles.dtype), control, target, self.qubits)
                state = apply_cry(state, self.coupling[edge+1].to(angles.dtype), target, control, self.qubits)
                edge += 2
        return state

    def forward(self, x):
        dtype = torch.float64 if self.input_proj.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(x.device.type, enabled=False):
            code = torch.tanh(F.linear(normalize_input(self.norm, x, dtype),
                                      self.input_proj.weight.to(dtype)))
            amplitudes = 1.0 + 0.5*torch.tanh(F.linear(code, self.amplitude_proj.weight.to(dtype)))
            state = F.normalize(amplitudes, dim=-1)
            angles = math.pi*torch.tanh(F.linear(code, self.angle_proj.weight.to(dtype)))
            state = self.evolve(state, angles.reshape(*x.shape[:-1], self.depth, self.qubits))
            probs = state.square()
            probs = probs/probs.sum(-1, keepdim=True)
            return ((probs + self.prob_floor)/(1+self.num_experts*self.prob_floor)).log()
