from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..gate_benchmark.models import MoTELinear, routed_modules, shared_state, load_shared


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CausalAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(b, length, 3, self.heads, dim // self.heads).unbind(2)
        q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
        # Causality is explicit. Right padding cannot affect any earlier position.
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.proj(a.transpose(1, 2).reshape(b, length, dim))


class Block(nn.Module):
    def __init__(self, cfg, variant, seed):
        super().__init__()
        m = cfg['model']
        self.norm1 = nn.LayerNorm(m['dim'])
        self.norm2 = nn.LayerNorm(m['dim'])
        self.attention = CausalAttention(m['dim'], m['heads'])
        self.up = MoTELinear(m['dim'], m['ff_dim'], cfg['router'], variant, seed)
        self.down = MoTELinear(m['ff_dim'], m['dim'], cfg['router'], variant, seed + 1)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class CausalLM(nn.Module):
    def __init__(self, cfg, variant, seed):
        super().__init__()
        m = cfg['model']
        self.context = m['context']
        self.embedding = nn.Embedding(m['vocab_size'], m['dim'])
        self.position = nn.Embedding(self.context, m['dim'])
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)
        self.blocks = nn.ModuleList([
            Block(cfg, variant, seed + 2*i) for i in range(m['layers'])
        ])
        self.norm = nn.LayerNorm(m['dim'])

    def forward(self, input_ids, valid_mask=None):
        if input_ids.shape[1] > self.context:
            raise ValueError('input exceeds model context')
        modules = routed_modules(self)
        for _, module in modules:
            module.attention_mask = valid_mask
        try:
            x = self.embedding(input_ids) + self.position(
                torch.arange(input_ids.shape[1], device=input_ids.device))
            for block in self.blocks:
                x = block(x)
            # Weight tying reduces the vocabulary head; identical for every router.
            return F.linear(self.norm(x), self.embedding.weight)
        finally:
            for _, module in modules:
                module.attention_mask = None


def initialized_model(cfg, variant, seed):
    """Anchor every non-router parameter/buffer to the full-MLP initialization."""
    seed_all(seed)
    anchor = CausalLM(cfg, 'mlp', seed)
    state = shared_state(anchor)
    if variant == 'mlp':
        return anchor, state
    del anchor
    seed_all(seed)
    model = CausalLM(cfg, variant, seed)
    load_shared(model, state)
    return model, state
