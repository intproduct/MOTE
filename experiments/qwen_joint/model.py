"""Pretrained dense conversion and genuinely sparse ADTN projection adapters."""
from dataclasses import asdict, dataclass
import hashlib
import itertools
import math
import random

import torch
from torch import nn

from ...ADTN import _block_contraction_plan, apply_block_to_sites
from .routers import JointRouter


@dataclass(frozen=True)
class AdapterConfig:
    seed: int = 17
    experts: int = 16
    top_k: int = 4
    feature_rank: int = 8
    # Leave two input sites uncontracted (kin=8 for hidden 1024).
    free_input_sites: int = 2
    residual_rank: int = 64
    tail_rank: int = 128


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(f"{name}:{t.dtype}:{tuple(t.shape)}\n".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _operator_axes(qin, kin, kout, positions):
    _, back = _block_contraction_plan(qin, kin, kout, tuple(positions))
    free = [i for i in range(qin) if i not in positions]
    qout = qin - kin + kout
    # Canonical dense axes: output sites, then input sites.
    axes = [back.index(i + 1) - 1 for i in range(len(free))]
    axes += [qout + i for i in free]
    axes += [qout + i for i in sorted(positions)]
    axes += [back.index(len(free) + i + 1) - 1 for i in range(kout)]
    return axes, qout, len(free)


def project_dense(weight, qin, kin, kout, positions):
    """Frobenius orthogonal projection onto U contracted with untouched identities."""
    axes, qout, nfree = _operator_axes(qin, kin, kout, positions)
    view = weight.reshape([2] * (qin + qout)).permute(axes)
    view = view.reshape(2**nfree, 2**nfree, 2**kin, 2**kout)
    return view.diagonal(dim1=0, dim2=1).mean(-1).contiguous()


def dense_from_tensor(u, qin, kin, kout, positions):
    axes, qout, nfree = _operator_axes(qin, kin, kout, positions)
    eye = torch.eye(2**nfree, dtype=u.dtype, device=u.device)
    canonical = eye[:, :, None, None] * u.reshape(1, 1, 2**kin, 2**kout)
    inverse = [axes.index(i) for i in range(len(axes))]
    return canonical.reshape([2] * (qin + qout)).permute(inverse).reshape(2**qout, 2**qin)


class TensorExpert(nn.Module):
    def __init__(self, u, positions, qin, kin, kout):
        super().__init__()
        self.u = nn.Parameter(u.reshape([2] * (kin + kout)).clone())
        self.positions, self.qin, self.kin, self.kout = tuple(positions), qin, kin, kout
        self.last_token_count = 0

    def forward(self, x):
        self.last_token_count = x.shape[0]
        y = apply_block_to_sites(x.reshape(-1, *([2] * self.qin)), self.u,
                                 self.positions, d=2, q_in=self.qin,
                                 k_in=self.kin, k_out=self.kout)
        return y.reshape(x.shape[0], -1)


class LowRank(nn.Module):
    @classmethod
    def empty(cls, out_features, in_features, rank):
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        rank = min(rank, in_features, out_features)
        result.a = nn.Parameter(torch.empty(out_features, rank))
        result.b = nn.Parameter(torch.empty(rank, in_features))
        return result

    def __init__(self, weight, rank, seed):
        super().__init__()
        rank = min(rank, *weight.shape)
        if rank < 1:
            raise ValueError("Low rank must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if rank == min(weight.shape):
                u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            else:
                u, s, v = torch.svd_lowrank(weight, q=min(rank + 8, *weight.shape), niter=2)
                vh = v.T
        self.a = nn.Parameter((u[:, :rank] * s[:rank]).contiguous())
        self.b = nn.Parameter(vh[:rank].contiguous())

    def forward(self, x):
        return torch.nn.functional.linear(torch.nn.functional.linear(x, self.b), self.a)

    def dense(self):
        return self.a @ self.b


class JointProjection(nn.Module):
    @classmethod
    def from_spec(cls, in_features, out_features, method, cfg, salt, conversion):
        """Allocate an adapter to receive a verified frozen common initialization."""
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.in_features, result.out_features = in_features, out_features
        result.qin, result.qout = in_features.bit_length() - 1, out_features.bit_length() - 1
        result.main_in, result.main_out = 2**result.qin, 2**result.qout
        kin, kout = conversion["kin"], conversion["kout"]
        result.experts = nn.ModuleList([TensorExpert(torch.empty(2**kin, 2**kout), pos,
            result.qin, kin, kout) for pos in conversion["positions"]])
        result.residual = LowRank.empty(result.main_out, result.main_in, cfg.residual_rank)
        result.tail01 = result.tail10 = result.tail11 = None
        if in_features > result.main_in:
            result.tail01 = LowRank.empty(result.main_out, in_features - result.main_in, cfg.tail_rank)
        if out_features > result.main_out:
            result.tail10 = LowRank.empty(out_features - result.main_out, result.main_in, cfg.tail_rank)
            if in_features > result.main_in:
                result.tail11 = LowRank.empty(out_features - result.main_out, in_features - result.main_in, cfg.tail_rank)
        result.router = JointRouter(in_features, method, cfg.seed + salt + 900000, cfg.feature_rank, cfg.top_k)
        result.aux_loss, result.last_metrics, result.capture_metrics = None, {}, False
        result.conversion = conversion
        return result

    def __init__(self, linear: nn.Linear, method: str, cfg: AdapterConfig, salt: int):
        super().__init__()
        if linear.bias is not None or cfg.experts != 16:
            raise ValueError("This release supports bias-free Qwen projections and 16 experts")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.qin = self.in_features.bit_length() - 1
        self.qout = self.out_features.bit_length() - 1
        self.main_in, self.main_out = 2**self.qin, 2**self.qout
        kin = self.qin - cfg.free_input_sites
        kout = kin + self.qout - self.qin
        if min(kin, kout) <= 0:
            raise ValueError("Too few tensor sites")
        positions = list(itertools.combinations(range(self.qin), kin))
        random.Random(cfg.seed * 100003 + salt).shuffle(positions)
        if len(positions) < cfg.experts:
            raise ValueError("Not enough distinct expert positions")
        positions = positions[:cfg.experts]
        weight = linear.weight.detach().float().cpu()
        w00 = weight[:self.main_out, :self.main_in]
        self.experts = nn.ModuleList()
        mean_operator = torch.zeros_like(w00)
        for pos in positions:
            u = project_dense(w00, self.qin, kin, kout, pos)
            self.experts.append(TensorExpert(u, pos, self.qin, kin, kout))
            mean_operator.add_(dense_from_tensor(u, self.qin, kin, kout, pos), alpha=1 / cfg.experts)
        self.residual = LowRank(w00 - mean_operator, cfg.residual_rank, cfg.seed + salt)
        self.tail01 = self.tail10 = self.tail11 = None
        if self.in_features > self.main_in:
            self.tail01 = LowRank(weight[:self.main_out, self.main_in:], cfg.tail_rank, cfg.seed + salt + 1)
        if self.out_features > self.main_out:
            self.tail10 = LowRank(weight[self.main_out:, :self.main_in], cfg.tail_rank, cfg.seed + salt + 2)
            if self.in_features > self.main_in:
                self.tail11 = LowRank(weight[self.main_out:, self.main_in:], cfg.tail_rank, cfg.seed + salt + 3)
        self.router = JointRouter(self.in_features, method, cfg.seed + salt + 900000,
                                  cfg.feature_rank, cfg.top_k)
        self.aux_loss = None
        self.last_metrics = {}
        self.capture_metrics = False
        with torch.no_grad():
            reconstructed = torch.zeros_like(weight)
            reconstructed[:self.main_out, :self.main_in] = mean_operator + self.residual.dense()
            for name, rows, cols in (("tail01", slice(None, self.main_out), slice(self.main_in, None)),
                                     ("tail10", slice(self.main_out, None), slice(None, self.main_in)),
                                     ("tail11", slice(self.main_out, None), slice(self.main_in, None))):
                part = getattr(self, name)
                if part is not None:
                    reconstructed[rows, cols] = part.dense()
            self.conversion = {"uniform_dense_relative_frobenius_error": float((weight - reconstructed).norm() / weight.norm()),
                               "positions": positions, "kin": kin, "kout": kout,
                               "expert_hash": tensor_hash({k: v for k, v in self.state_dict().items() if not k.startswith("router.")}),
                               "distinct_expert_hashes": len({tensor_hash({"u": e.u}) for e in self.experts})}

    def set_precision(self, dtype, device):
        self.to(device=device, dtype=dtype)
        self.router.float()
        return self

    def forward(self, inputs):
        shape = inputs.shape[:-1]
        x = inputs.reshape(-1, self.in_features)
        route = self.router(x)
        self.aux_loss = route.load_balance
        main = x[:, :self.main_in]
        aggregate = torch.zeros(len(x), self.main_out, device=x.device, dtype=torch.float32)
        # Only tokens actually assigned to an expert enter that expert's contraction.
        for eid, expert in enumerate(self.experts):
            token, slot = torch.where(route.indices == eid)
            expert.last_token_count = token.numel()
            if token.numel():
                contribution = expert(main.index_select(0, token)).float()
                aggregate.index_add_(0, token, contribution * route.weights[token, slot, None])
        ymain = aggregate.to(inputs.dtype) + self.residual(main)
        if self.tail01 is not None:
            ymain = ymain + self.tail01(x[:, self.main_in:])
        if self.tail10 is not None:
            ytail = self.tail10(main)
            if self.tail11 is not None:
                ytail = ytail + self.tail11(x[:, self.main_in:])
            ymain = torch.cat((ymain, ytail), dim=-1)
        if self.capture_metrics:
            with torch.no_grad():
                load = torch.bincount(route.indices.flatten(), minlength=16).float()
                p = route.probabilities
                self.last_metrics = {"entropy": float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean()),
                                     "load_cv": float(load.std(unbiased=False) / load.mean()),
                                     "unused_experts": int((load == 0).sum()),
                                     "dispatch_tokens": [e.last_token_count for e in self.experts]}
        return ymain.reshape(*shape, self.out_features)


def patch_qwen(model, method: str, cfg: AdapterConfig, common=None):
    """Freeze backbone and adapt the last ceil-quarter FFNs, or matched dense FFNs."""
    model.requires_grad_(False)
    layers = model.model.layers
    selected = list(range(3 * len(layers) // 4, len(layers)))
    if common is not None and (common["adapter"] != asdict(cfg) or common["layers"] != selected):
        raise ValueError("Common initialization configuration/layers differ")
    report = {"method": method, "layers": selected, "adapter": asdict(cfg), "projections": {}}
    for index in selected:
        mlp = layers[index].mlp
        if method == "dense":
            mlp.requires_grad_(True)
            continue
        for offset, name in enumerate(("gate_proj", "up_proj", "down_proj")):
            original = getattr(mlp, name)
            salt = index * 100 + offset * 10
            if common is None:
                projection = JointProjection(original, method, cfg, salt)
            else:
                projection = JointProjection.from_spec(original.in_features, original.out_features, method, cfg, salt,
                    common["projections"][f"{index}.{name}"])
            setattr(mlp, name, projection)
            report["projections"][f"{index}.{name}"] = projection.conversion
    if common is not None and method != "dense":
        expected = {n for n, p in model.named_parameters() if p.requires_grad and ".router." not in n}
        if set(common["state"]) != expected or tensor_hash(common["state"]) != common["state_hash"]:
            raise ValueError("Common initialization parameter names or hash differ")
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in expected:
                    p.copy_(common["state"][name])
    counts = {"total": sum(p.numel() for p in model.parameters()),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    for group, fragment in (("router", ".router."), ("expert", ".experts.")):
        counts[group] = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and fragment in n)
    counts["residual_tail_or_dense"] = counts["trainable"] - counts["router"] - counts["expert"]
    report["parameters"] = counts
    return report


def set_model_precision(model, dtype, device):
    model.to(device=device, dtype=dtype)
    for module in model.modules():
        if isinstance(module, JointRouter):
            module.float()


def auxiliary_loss(model):
    losses = [m.aux_loss for m in model.modules() if isinstance(m, JointProjection) and m.aux_loss is not None]
    return torch.stack(losses).mean() if losses else next(model.parameters()).new_zeros((), dtype=torch.float32)
