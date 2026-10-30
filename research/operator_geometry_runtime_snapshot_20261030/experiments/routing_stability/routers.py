from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from ...gate import TopKGate
from ...quantum_gate import apply_ry, apply_cry
from ...structured_routers import normalize_input


def standardized_scores(scores, eps=1e-6):
    centered = scores - scores.mean(-1, keepdim=True)
    # The scale participates in autograd: gain alone cannot buy a larger gap.
    return centered / (centered.square().mean(-1, keepdim=True) + eps**2).sqrt()


def boundary_loss(scores, k, threshold, eps=1e-6):
    if not 0 < k < scores.shape[-1] or threshold <= 0:
        raise ValueError('require 0 < k < E and positive threshold')
    values = standardized_scores(scores, eps).topk(k+1, dim=-1).values
    gap = values[..., k-1] - values[..., k]
    return (threshold-gap).clamp_min(0).square().mean(), gap


class SharedBornRouter(nn.Module):
    """Bounded joint encoding, shared orthogonal RY/CRY, direct Born readout.

    States have E=2**q amplitudes, NOT q amplitudes. U is independent of input
    but learnable. A chain of q-1 controlled rotations connects every bit cut.
    The dense E x E matrix is contracted on E basis states, not on every token.
    In eval/no-grad it is cached, with parameter versions/device/dtype as key.
    No angle re-upload, output gain, temperature exponent, or log-softmax.
    """
    gate_arch = 'shared_born'

    def __init__(self, data_dim, experts, rank, output_init_std=1e-3,
                 prob_floor=1e-6, entangle=True):
        super().__init__()
        if experts < 2 or experts > 64 or experts & (experts-1) or rank < 1:
            raise ValueError('E must be a power of two in [2,64]; rank >= 1')
        if not 0 < prob_floor < 1/experts:
            raise ValueError('probability floor must be in (0,1/E)')
        self.data_dim, self.num_experts, self.rank = data_dim, experts, rank
        self.qubits = experts.bit_length()-1
        self.prob_floor = prob_floor
        self.norm = nn.LayerNorm(data_dim)
        self.input_proj = nn.Linear(data_dim, rank, bias=False)
        self.amplitude_proj = nn.Linear(rank, experts, bias=False)
        nn.init.normal_(self.amplitude_proj.weight, std=output_init_std)
        self.rotation = nn.Parameter(torch.zeros(self.qubits))
        self.coupling = nn.Parameter(torch.zeros(self.qubits-1)) if entangle else None
        self._matrix_cache = None
        self._cache_key = None

    @property
    def router_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def evolve(self, state):
        for wire in range(self.qubits):
            state = apply_ry(state, self.rotation[wire].to(state.dtype), wire, self.qubits)
        if self.coupling is not None:
            for wire in range(self.qubits-1):
                state = apply_cry(state, self.coupling[wire].to(state.dtype), wire, wire+1, self.qubits)
        return state

    def matrix(self, dtype):
        # Evolving identity rows returns U.T: state row vectors right-multiply it.
        key = (self.rotation._version,
               self.coupling._version if self.coupling is not None else None,
               self.rotation.device, dtype)
        use_cache = not self.training and not torch.is_grad_enabled()
        if use_cache and self._cache_key == key and self._matrix_cache is not None:
            return self._matrix_cache
        value = self.evolve(torch.eye(self.num_experts, device=self.rotation.device, dtype=dtype))
        if use_cache:
            self._matrix_cache, self._cache_key = value, key
        return value

    def forward(self, x):
        dtype = torch.float64 if self.input_proj.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(x.device.type, enabled=False):
            code = torch.tanh(F.linear(normalize_input(self.norm, x, dtype), self.input_proj.weight.to(dtype)))
            amplitude = 1 + .5 * torch.tanh(F.linear(code, self.amplitude_proj.weight.to(dtype)))
            state = F.normalize(amplitude, dim=-1) @ self.matrix(dtype)
            p = state.square()
            p = p / p.sum(-1, keepdim=True)
            return (p+self.prob_floor)/(1+self.num_experts*self.prob_floor)


class ProbabilityTopKGate(TopKGate):
    """Reuse capacity, hard weights, STE, and balancing from the existing gate.

    The router already returns probabilities. This equals log(p)->softmax only
    at gain/temperature=1; the original Born's learnable gain is NOT retained.
    """
    def __init__(self, router, k, aux_coeff):
        super().__init__(router.data_dim, router.num_experts, k=k,
                         temperature=1., jitter_eps=0., drop_tokens=False,
                         use_ste=True, aux_coeff=aux_coeff, zloss_coeff=0.)
        self.router = router

    def _softmax_temp(self, probabilities):
        if self.temperature != 1 or self.jitter_eps or self.zloss_coeff or self.drop_tokens:
            raise ValueError('direct Born requires T=1, no jitter/z-loss/token dropping')
        return probabilities
