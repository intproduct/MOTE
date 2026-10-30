from __future__ import annotations

import hashlib
import torch
from ..autoregressive_gate.model import CausalLM as OriginalLM, seed_all
from ..gate_benchmark.models import routed_modules, shared_state, load_shared
from .routers import SharedBornRouter, ProbabilityTopKGate, boundary_loss


def full_state_hash(state):
    """Include scalar router gains; legacy shared-state hash sees no scalars."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class CausalLM(OriginalLM):
    def __init__(self, cfg, variant, seed):
        base = cfg['designs'][variant]['base']
        super().__init__(cfg, 'mlp_half' if base == 'born_lite' else base, seed)
        self.margin_enabled = cfg['designs'][variant]['margin']
        self.margin_cfg = cfg['margin']
        self.total_steps = cfg['train']['steps']
        self.training_step = 0
        self._margin_handles = []
        self._margin_terms = []
        if base == 'born_lite':
            for index, (_, module) in enumerate(routed_modules(self)):
                previous = module.mote.core.gate
                budget = sum(p.numel() for p in previous.router.parameters())
                dim, experts = module.in_features, cfg['router']['experts']
                q = experts.bit_length()-1
                rank = max(1, round((budget - 2*dim - (2*q-1))/(dim+experts)))
                # Each router gets a reproducible independent initialization.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed+index+700000)
                    router = SharedBornRouter(dim, experts, rank,
                                              cfg['router']['output_init_std'])
                    module.mote.core.gate = ProbabilityTopKGate(router, previous.k, previous.aux_coeff)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if not hasattr(self, '_margin_handles'):
            return self
        if not mode:
            for handle in self._margin_handles:
                handle.remove()
            self._margin_handles.clear()
            self._margin_terms.clear()
        elif self.margin_enabled and not self._margin_handles:
            for _, module in routed_modules(self):
                gate = module.mote.core.gate
                def capture(router, inputs, output, k=gate.k):
                    if self.margin_weight() > 0:
                        # Log probabilities align score semantics with original Born.
                        score = output.log() if isinstance(router, SharedBornRouter) else output
                        term, gap = boundary_loss(score, k, self.margin_cfg['threshold'], self.margin_cfg['eps'])
                        self._margin_terms.append((term, gap.detach()))
                self._margin_handles.append(gate.router.register_forward_hook(capture))
        return self

    def margin_weight(self):
        if not self.training or not self.margin_enabled:
            return 0.
        start = self.margin_cfg['start_fraction'] * self.total_steps
        end = self.margin_cfg['full_fraction'] * self.total_steps
        return self.margin_cfg['weight'] * max(0., min(1., (self.training_step-start)/max(end-start, 1)))

    def forward(self, input_ids, valid_mask=None):
        self._margin_terms.clear()
        return super().forward(input_ids, valid_mask)

    def regularizer(self):
        if not self._margin_terms:
            return self.embedding.weight.new_zeros(())
        return self.margin_weight() * torch.stack([p[0] for p in self._margin_terms]).mean()

    def margin_metrics(self):
        if not self._margin_terms:
            return {'margin_weight': self.margin_weight(), 'margin_loss': 0.}
        gap = torch.cat([p[1].flatten() for p in self._margin_terms])
        return {'margin_weight': self.margin_weight(), 'margin_loss': float(self.regularizer().detach()),
                'standardized_gap_mean': float(gap.mean()),
                'standardized_gap_p05': float(torch.quantile(gap, .05)),
                'fraction_below_threshold': float((gap < self.margin_cfg['threshold']).float().mean())}


def initialized_model(cfg, variant, seed):
    seed_all(seed)
    anchor = OriginalLM(cfg, 'mlp', seed)
    state = shared_state(anchor)
    del anchor
    seed_all(seed)
    model = CausalLM(cfg, variant, seed)
    load_shared(model, state)
    return model, state
