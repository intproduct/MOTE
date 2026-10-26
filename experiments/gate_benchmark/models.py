from __future__ import annotations
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from ...sparse_mixt import SparseMiXTLinear


def router_config(c, arch, dim, seed):
    e, q, depth = c["experts"], int(math.log2(c["experts"])), c["depth"]
    target = (
        dim * c["sites"] + c["sites"] * c["bond_dim"] ** 2 + c["bond_dim"] * (e + 1)
    )
    edges = sum(2 * len(range(i % 2, q - 1, 2)) for i in range(depth))
    edges = edges if c["quantum_entangle"] else 0
    hidden = (
        max(1, round((target - e) / (dim + e + 1)))
        if c["budget"] == "matched"
        else c["rank"]
    )
    rank = (
        max(1, round((target - 2 * depth * q - edges - q) / (dim + 2 * depth * q)))
        if c["budget"] == "matched"
        else c["rank"]
    )
    return dict(
        E=e,
        d=2,
        k_in=c["k_in"],
        seed=seed,
        pos_strategy="random",
        gate_type="topk",
        topk=c["topk"],
        gate_arch=arch,
        gate_hidden_dim=hidden,
        gate_norm=c["norm"],
        gate_dropout=0.0,
        gate_activation="silu",
        gate_output_init_std=c["output_init_std"],
        gate_learnable_gain=c["learnable_gain"],
        gate_quantum=dict(rank=rank, depth=depth, entangle=c["quantum_entangle"]),
        gate_mps=dict(sites=c["sites"], bond_dim=c["bond_dim"]),
        temperature=c["temperature"],
        aux_coeff=c["aux_coeff"],
        zloss_coeff=0.0,
        drop_tokens=c["drop_tokens"],
        use_ste=True,
    )


class MoTELinear(nn.Module):
    """Tensor-only FFN interface, excluding padding from routing AND aux loss."""

    def __init__(self, din, dout, cfg, arch, seed, bias=None, source_weight=None):
        super().__init__()
        self.in_features, self.out_features = din, dout
        self.mote = SparseMiXTLinear(
            din,
            dout,
            cfg=router_config(cfg, arch, din, seed),
            ranks={"01": 1, "10": 1, "11": 1},
        )
        if self.mote.core.num_blocks != cfg["experts"]:
            raise ValueError(
                "Not enough unique tensor positions for requested E; increase width or change k_in"
            )
        self.bias = nn.Parameter(
            torch.zeros(dout) if bias is None else bias.detach().clone()
        )
        self.register_buffer(
            "source_weight",
            None if source_weight is None else source_weight.detach().clone(),
            persistent=False,
        )
        self.attention_mask = None
        self.last_aux = None
        self.last_tokens = 0

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        valid = None
        if self.attention_mask is not None:
            valid = self.attention_mask.reshape(-1).bool()
            if valid.numel() != flat.shape[0]:
                raise ValueError("attention mask and token count differ")
            flat = flat[valid]
        self.last_tokens = flat.shape[0]
        if not flat.shape[0]:
            self.last_aux = None
            return x.new_zeros(*shape[:-1], self.out_features) + self.bias
        y, self.last_aux = self.mote(flat, return_aux=True)
        y = y + self.bias
        if valid is not None:
            padded = y.new_zeros(valid.numel(), self.out_features)
            y = padded.index_copy(0, valid.nonzero(as_tuple=True)[0], y)
        return y.reshape(*shape[:-1], self.out_features)


def routed_modules(model):
    return [(name, m) for name, m in model.named_modules() if isinstance(m, MoTELinear)]


def auxiliary_loss(model):
    losses = [
        m.last_aux["l_aux"] for _, m in routed_modules(model) if m.last_aux is not None
    ]
    return (
        torch.stack(losses).mean() if losses else next(model.parameters()).new_zeros(())
    )


def routing_snapshot(model):
    result = {}
    for name, module in routed_modules(model):
        if module.last_aux is None:
            continue
        result[name] = {
            k: v.detach().float().cpu().tolist()
            for k, v in module.last_aux.items()
            if k in {"H_tok", "H_batch", "importance", "load", "drop_rate", "l_aux"}
        }
        result[name]["tokens"] = module.last_tokens
    return result


class ImageBlock(nn.Module):
    def __init__(self, cfg, arch, seed):
        super().__init__()
        w, h = cfg["mlp"]["width"], cfg["mlp"]["width"] * cfg["mlp"]["expansion"]
        self.norm = nn.LayerNorm(w)
        self.mixer = nn.Linear(w, w)
        self.up = MoTELinear(w, h, cfg, arch, seed)
        self.down = MoTELinear(h, w, cfg, arch, seed + 1)

    def forward(self, x):
        h = F.gelu(self.mixer(self.norm(x)))
        return x + 0.1 * self.down(F.gelu(self.up(h)))


class ImageClassifier(nn.Module):
    def __init__(self, cfg, arch, seed, input_dim, classes):
        super().__init__()
        w = cfg["mlp"]["width"]
        self.stem = nn.Linear(input_dim, w)
        self.blocks = nn.ModuleList(
            [ImageBlock(cfg, arch, seed + 2 * i) for i in range(cfg["mlp"]["layers"])]
        )
        self.head = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, classes))

    def forward(self, features, **unused):
        x = F.gelu(self.stem(features))
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class BertClassifier(nn.Module):
    def __init__(self, cfg, arch, seed, classes, model_path=None, tiny_config=None):
        super().__init__()
        from transformers import BertModel

        if tiny_config is not None:
            self.bert = BertModel(tiny_config)
        else:
            self.bert = BertModel.from_pretrained(
                model_path, local_files_only=True, attn_implementation="eager"
            )
        layers = cfg["bert_layers"]
        if len(self.bert.encoder.layer) < layers:
            raise ValueError("pretrained BERT has too few layers")
        self.bert.encoder.layer = nn.ModuleList(list(self.bert.encoder.layer)[:layers])
        self.bert.config.num_hidden_layers = layers
        for i, layer in enumerate(self.bert.encoder.layer):
            for j, owner in enumerate((layer.intermediate, layer.output)):
                old = owner.dense
                owner.dense = MoTELinear(
                    old.in_features,
                    old.out_features,
                    cfg,
                    arch,
                    seed + 2 * i + j,
                    old.bias,
                    old.weight,
                )
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **unused):
        modules = routed_modules(self)
        for _, module in modules:
            module.attention_mask = attention_mask
        try:
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return self.classifier(self.dropout(out.pooler_output))
        finally:
            for _, module in modules:
                module.attention_mask = None


def is_router_key(key):
    return ".core.gate." in key


def shared_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not is_router_key(k)
    }


def load_shared(model, state):
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not is_router_key(k) for k in result.missing_keys):
        raise RuntimeError(f"shared initialization mismatch: {result}")


def state_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(str((v.dtype, tuple(v.shape))).encode())
        h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def parameter_report(model):
    routers = {
        name: sum(p.numel() for p in m.mote.core.gate.router.parameters())
        for name, m in routed_modules(model)
    }
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "router_total": sum(routers.values()),
        "routers": routers,
        "positions": {n: m.mote.core.positions for n, m in routed_modules(model)},
    }


def warm_initialize(model, steps, seed):
    """Common expert-only linear-operator fit, using independent Gaussian probes.

    This fits the uniform all-expert average, not a routed teacher output.
    Report held-out probe error so approximation loss stays visible.
    """
    reports = {}
    for i, (name, m) in enumerate(routed_modules(model)):
        if m.source_weight is None or steps == 0:
            continue
        device = m.source_weight.device
        g = torch.Generator(device=device).manual_seed(seed + i)
        probe = torch.randn(64, m.in_features, device=device, generator=g)
        target = F.linear(probe, m.source_weight)

        def error():
            with torch.no_grad():
                y, _ = m.mote(probe, dense=True)
                return float(
                    (y - target).square().mean()
                    / target.square().mean().clamp_min(1e-12)
                )

        before = error()
        optimizer = torch.optim.AdamW(
            [p for n, p in m.mote.named_parameters() if not is_router_key("." + n)],
            lr=0.01,
            weight_decay=0,
        )
        for _ in range(steps):
            x = torch.randn(64, m.in_features, device=device, generator=g)
            y, _ = m.mote(x, dense=True)
            loss = F.mse_loss(y, F.linear(x, m.source_weight))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        reports[name] = {
            "relative_mse_before": before,
            "relative_mse_after": error(),
            "steps": steps,
        }
    # The teacher matrices are not needed during task training.
    for _, m in routed_modules(model):
        m.source_weight = None
    return reports
