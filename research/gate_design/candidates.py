"""Research-only router candidates: exact local circuits and a classical MPS.

These return logits and can be assigned to TopKGate.router in experiments.
They are deliberately not registered in the production config/checkpoint schema.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from fitmotn.quantum_gate import apply_ry, apply_cry


class AngleRouter(nn.Module):
    def __init__(self, data_dim, num_experts, rank=16, readout="expectation"):
        super().__init__()
        if min(data_dim, num_experts, rank) < 1:
            raise ValueError("dimensions must be positive")
        if readout not in {"expectation", "born"}:
            raise ValueError("unknown readout")
        self.data_dim, self.num_experts = data_dim, num_experts
        self.readout = readout
        self.input_proj = nn.Linear(data_dim, rank, bias=False)
        self.angle_proj = nn.Linear(rank, num_experts, bias=False)
        nn.init.normal_(self.angle_proj.weight, std=1e-3)
        self.raw_gain = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def encode(self, x, dtype):
        code = F.linear(x.to(dtype), self.input_proj.weight.to(dtype))
        return math.pi * torch.tanh(F.linear(code, self.angle_proj.weight.to(dtype)))

    def forward(self, x):
        dtype = torch.float64 if self.input_proj.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(x.device.type, enabled=False):
            z = self.expectations_from_angles(self.encode(x, dtype))
            if self.readout == "born":
                logits = (1e-4 + (1 - 2e-4) * ((1 - z) / 2).clamp(0, 1)).log()
            else:
                logits = -z
            return F.softplus(self.raw_gain.to(dtype)) * logits


def brick_schedule(experts, depth):
    """Each layer = data rotations + ONE alternating nearest-neighbour matching."""
    operations, edge_count = [], 0
    for layer in range(depth):
        operations.extend(("ry", layer, q) for q in range(experts))
        for q in range(layer % 2, experts - 1, 2):
            control, target = (q, q + 1) if layer % 2 == 0 else (q + 1, q)
            for c, t in ((control, target), (target, control)):
                operations.append(("cry", edge_count, c, t))
                edge_count += 1
    operations.extend(("readout", q) for q in range(experts))
    return operations, edge_count


class _ConeGroup(nn.Module):
    def __init__(self, width, target, program, indices, outputs):
        super().__init__()
        self.width, self.target, self.program = width, target, program
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long).T.contiguous())
        self.register_buffer("outputs", torch.tensor(outputs, dtype=torch.long))
        basis = torch.arange(2**width)
        self.register_buffer("signs", 1 - 2 * ((basis >> (width - target - 1)) & 1))


class BrickwallConeGate(AngleRouter):
    """Connected shallow VQC, evaluated by exact batched causal-cone contraction.

    A depth-L nearest-neighbour matching circuit has at most 2L sites in each
    local readout cone (for L>=1). No MPS truncation or full-register state.
    Internal cones with identical schedules share one vectorized contraction.
    """
    def __init__(self, data_dim, num_experts, rank=16, depth=2, readout="expectation"):
        super().__init__(data_dim, num_experts, rank, readout)
        if depth not in {1, 2, 3, 4}:
            raise ValueError("research cone depth must be 1, 2, 3 or 4")
        self.depth = depth
        self.operations, edges = brick_schedule(num_experts, depth)
        self.encoding_scale = nn.Parameter(torch.ones(depth, num_experts))
        self.rotation = nn.Parameter(torch.empty(depth, num_experts).uniform_(-0.1, 0.1))
        self.coupling = nn.Parameter(torch.full((edges,), 0.2))
        self.readout_rotation = nn.Parameter(torch.full((num_experts,), math.pi / 2))
        grouped = defaultdict(list)
        for output in range(num_experts):
            active, kept = {output}, []
            for op in reversed(self.operations):
                wires = op[2:] if op[0] == "cry" else (op[-1],)
                if active.intersection(wires):
                    active.update(wires)
                    kept.append(op)
            mapping = {q: i for i, q in enumerate(sorted(active))}
            program, indices = [], []
            for op in reversed(kept):
                if op[0] == "ry":
                    program.append(("ry", op[1], mapping[op[2]]))
                    indices.append(op[2])
                elif op[0] == "cry":
                    program.append(("cry", mapping[op[2]], mapping[op[3]]))
                    indices.append(op[1])
                else:
                    program.append(("readout", mapping[op[1]]))
                    indices.append(op[1])
            grouped[(len(active), mapping[output], tuple(program))].append((output, indices))
        self.cones = nn.ModuleList()
        concatenated = []
        for (width, target, program), values in grouped.items():
            outputs, indices = zip(*values)
            self.cones.append(_ConeGroup(width, target, program, indices, outputs))
            concatenated.extend(outputs)
        self.register_buffer("restore_order", torch.tensor(concatenated).argsort())
        self.max_cone_qubits = max(c.width for c in self.cones)

    def expectations_from_angles(self, angles):
        shape, dtype = angles.shape, angles.dtype
        a = angles.reshape(-1, self.num_experts)
        theta = a[:, None, :] * self.encoding_scale.to(dtype) + self.rotation.to(dtype)
        gamma, beta = self.coupling.to(dtype), self.readout_rotation.to(dtype)
        outputs = []
        for cone in self.cones:
            state = a.new_zeros((a.shape[0], cone.outputs.numel(), 2**cone.width))
            state[..., 0] = 1
            for index, op in zip(cone.indices, cone.program):
                if op[0] == "ry":
                    state = apply_ry(state, theta[:, op[1], index], op[2], cone.width)
                elif op[0] == "cry":
                    state = apply_cry(state, gamma[index], op[1], op[2], cone.width)
                else:
                    state = apply_ry(state, beta[index], op[1], cone.width)
            outputs.append((state.square() * cone.signs.to(dtype)).sum(-1))
        return torch.cat(outputs, -1).index_select(-1, self.restore_order).reshape(shape)


def cry_matrix(angle, reverse=False):
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    rows = ((one, zero, zero, zero), (zero, c, zero, -s),
            (zero, zero, one, zero), (zero, s, zero, c)) if reverse else (
            (one, zero, zero, zero), (zero, one, zero, zero),
            (zero, zero, c, -s), (zero, zero, s, c))
    return torch.stack([torch.stack(row, -1) for row in rows], -2)


class SweepDensityGate(AngleRouter):
    """Two independent opposite sweeps, with averaged local expectations.

    Each pure-state sweep is an MPS with bond dimension <=2. All one-site
    readouts are exact using a 2x2 frontier density matrix and a 4x4 two-site
    density matrix. A site never interacts again after its readout is computed.
    The average is a classical mixture of TWO circuits, not a two-sweep circuit.
    """
    def __init__(self, data_dim, num_experts, rank=16, bidirectional=True, readout="expectation"):
        super().__init__(data_dim, num_experts, rank, readout)
        self.directions = 2 if bidirectional else 1
        self.rotation = nn.Parameter(torch.empty(self.directions, num_experts).uniform_(-0.1, 0.1))
        # Both CRY directions within each two-site gate transmit more than just
        # the control populations; this is not a one-direction CRY-only channel.
        self.coupling = nn.Parameter(torch.full((self.directions, num_experts - 1, 2), 0.2))
        self.readout_rotation = nn.Parameter(torch.full((self.directions, num_experts), math.pi / 2))

    @staticmethod
    def read_z(rho, beta):
        return beta.cos() * (rho[..., 0, 0] - rho[..., 1, 1]) - beta.sin() * (rho[..., 0, 1] + rho[..., 1, 0])

    def expectations_from_angles(self, angles):
        shape, dtype = angles.shape, angles.dtype
        a = angles.reshape(-1, self.num_experts)
        all_results = []
        for direction in range(self.directions):
            order = list(range(self.num_experts))
            if direction:
                order.reverse()
            theta = a + self.rotation[direction].to(dtype)
            phi = torch.stack(((theta / 2).cos(), (theta / 2).sin()), -1)
            local = phi.unsqueeze(-1) * phi.unsqueeze(-2)
            beta = self.readout_rotation[direction].to(dtype)
            gamma = self.coupling[direction].to(dtype)
            rho = local[:, order[0]]
            results = [None] * self.num_experts
            for q, next_q in zip(order[:-1], order[1:]):
                edge = min(q, next_q)
                unitary = cry_matrix(gamma[edge, 1], True) @ cry_matrix(gamma[edge, 0])
                joint = torch.einsum("nab,ncd->nacbd", rho, local[:, next_q]).reshape(-1, 4, 4)
                joint = unitary @ joint @ unitary.T
                four = joint.reshape(a.shape[0], 2, 2, 2, 2)
                old_site = four.diagonal(dim1=2, dim2=4).sum(-1)
                results[q] = self.read_z(old_site, beta[q])
                rho = four.diagonal(dim1=1, dim2=3).sum(-1)
            results[order[-1]] = self.read_z(rho, beta[order[-1]])
            all_results.append(torch.stack(results, -1))
        return torch.stack(all_results).mean(0).reshape(shape)


class ResidualMPSGate(nn.Module):
    """Classical normalized matrix-product features, NOT a unitary circuit.

    M_i(x_i)=I+tanh(x_i) A_i/sqrt(m); the boundary vector contracts left to
    right and an output matrix returns E logits. Intermediate normalization
    preserves the final normalized direction (except the numerical epsilon).
    Sites m and bond dimension chi are independent of expert count.
    """
    def __init__(self, data_dim, num_experts, sites=16, bond_dim=8):
        super().__init__()
        if min(data_dim, num_experts, sites, bond_dim) < 1:
            raise ValueError("dimensions must be positive")
        self.sites, self.bond_dim = sites, bond_dim
        self.input_proj = nn.Linear(data_dim, sites, bias=False)
        self.cores = nn.Parameter(torch.randn(sites, bond_dim, bond_dim) / math.sqrt(bond_dim))
        self.boundary = nn.Parameter(F.normalize(torch.randn(bond_dim), dim=0))
        self.output_proj = nn.Linear(bond_dim, num_experts, bias=False)
        nn.init.normal_(self.output_proj.weight, std=1e-3)
        self.raw_gain = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def forward(self, x):
        dtype = torch.float64 if self.input_proj.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(x.device.type, enabled=False):
            values = torch.tanh(F.linear(x.to(dtype), self.input_proj.weight.to(dtype)))
            state = self.boundary.to(dtype).expand(*values.shape[:-1], self.bond_dim)
            cores = self.cores.to(dtype)
            for i in range(self.sites):
                state = state + values[..., i, None] * (state @ cores[i]) / math.sqrt(self.sites)
                state = F.normalize(state, dim=-1, eps=1e-8)
            return F.softplus(self.raw_gain.to(dtype)) * F.linear(state, self.output_proj.weight.to(dtype))


def apply_rz(state, angle, wire, width):
    shape = state.shape
    split = state.reshape(*shape[:-1], 2**wire, 2, 2 ** (width - wire - 1))
    zero, one = split.unbind(-2)
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    phase0 = torch.complex(c, -s)[..., None, None]
    phase1 = torch.complex(c, s)[..., None, None]
    return torch.stack((phase0 * zero, phase1 * one), -2).reshape(shape)


class CompactBornGate(nn.Module):
    """ceil(log2(E)) qubits, one joint computational-basis outcome per expert.

    Full state size is <2E, not 2**E. Two data angles per qubit PER layer
    avoid compressing every token to only log2(E) input scalars. Non-power-of-
    two E uses the first E outcomes and conditions on this valid subset.
    No quantum hardware sampling or complexity advantage is claimed.
    """
    def __init__(self, data_dim, num_experts, rank=16, depth=2):
        super().__init__()
        if min(data_dim, rank) < 1 or num_experts < 2 or depth not in {1, 2, 3, 4}:
            raise ValueError("positive dimensions, E>=2 and depth in [1,4] required")
        self.num_experts, self.depth = num_experts, depth
        self.qubits = (num_experts - 1).bit_length()
        self.input_proj = nn.Linear(data_dim, rank, bias=False)
        self.angle_proj = nn.Linear(rank, 2 * depth * self.qubits, bias=False)
        nn.init.normal_(self.angle_proj.weight, std=1e-3)
        self.rotation = nn.Parameter(torch.empty(depth, self.qubits, 2).uniform_(-0.05, 0.05))
        with torch.no_grad():
            self.rotation[0, :, 0].add_(math.pi / 4)
        _, edges = brick_schedule(self.qubits, depth)
        self.coupling = nn.Parameter(torch.full((edges,), 0.05))
        self.readout_rotation = nn.Parameter(torch.full((self.qubits,), math.pi / 4))
        self.raw_gain = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def probabilities_from_angles(self, angles):
        """Return ALL 2**q probabilities, including invalid padded outcomes."""
        leading = angles.shape[:-3]
        a = angles.reshape(-1, self.depth, self.qubits, 2)
        complex_dtype = torch.complex128 if angles.dtype == torch.float64 else torch.complex64
        state = torch.zeros((a.shape[0], 2**self.qubits), device=a.device, dtype=complex_dtype)
        state[:, 0] = 1
        theta = a + self.rotation.to(angles.dtype)
        gamma = self.coupling.to(angles.dtype)
        edge = 0
        for layer in range(self.depth):
            for q in range(self.qubits):
                state = apply_ry(state, theta[:, layer, q, 0], q, self.qubits)
                state = apply_rz(state, theta[:, layer, q, 1], q, self.qubits)
            for q in range(layer % 2, self.qubits - 1, 2):
                c, t = (q, q + 1) if layer % 2 == 0 else (q + 1, q)
                state = apply_cry(state, gamma[edge], c, t, self.qubits)
                state = apply_cry(state, gamma[edge + 1], t, c, self.qubits)
                edge += 2
        for q in range(self.qubits):
            state = apply_ry(state, self.readout_rotation[q].to(angles.dtype), q, self.qubits)
        return (state.real.square() + state.imag.square()).reshape(*leading, 2**self.qubits)

    def forward(self, x):
        dtype = torch.float64 if self.input_proj.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(x.device.type, enabled=False):
            code = F.linear(x.to(dtype), self.input_proj.weight.to(dtype))
            angles = math.pi * torch.tanh(F.linear(code, self.angle_proj.weight.to(dtype)))
            angles = angles.reshape(*x.shape[:-1], self.depth, self.qubits, 2)
            full = self.probabilities_from_angles(angles)
            # Existing router softmax normalizes the valid subset. Smoothing
            # avoids log(0); monitor full[..., :E].sum(-1) in real experiments.
            logits = (full[..., :self.num_experts] + 1e-6).log()
            return F.softplus(self.raw_gain.to(dtype)) * logits
