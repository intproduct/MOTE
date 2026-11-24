"""Lossless import of trained hybrid MixT into the shared SparseMiXT runtime.

The serialized specification is sufficient to rebuild without the partner's
Python code. Weight import never materializes a dense approximation or an SVD.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from .ADTN import GatedADTNLayer
from .gate import gate_config_from_mapping, gate_factory_config
from .sparse_mixt import LowRankLinear, SparseMiXTLinear, resolve_main_dim

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def projection_spec(source) -> dict[str, Any]:
    if not getattr(source, "is_mixt_projector", False):
        raise TypeError("pretrained_mixt requires trained native MixT projections")
    main = source.mixt00
    spec = dict(in_features=int(source.in_features), out_features=int(source.out_features),
                main_in_features=int(source.main_in_features), main_out_features=int(source.main_out_features),
                d=int(main.d), k_in=int(main.k_in), k_out=int(main.k_out),
                positions=copy.deepcopy(main.op_indices), bias=source.bias is not None,
                branches={})
    for target, original in (("01", "10"), ("10", "01"), ("11", "11")):
        branch = getattr(source, "w" + original)
        if branch is None:
            spec["branches"][target] = None
        elif isinstance(branch, nn.Linear) and branch.bias is None:
            spec["branches"][target] = {"mode": "dense"}
        elif (isinstance(branch, nn.Sequential) and len(branch) == 2 and
              all(isinstance(m, nn.Linear) and m.bias is None for m in branch)):
            spec["branches"][target] = {"mode": "lowrank", "rank": int(branch[0].out_features)}
        else:
            raise TypeError(f"Unsupported native MixT branch w{original}: {type(branch)}")
    return spec


def resolve_import_config(model, layers, selected, cfg):
    native = copy.deepcopy(cfg.get("pretrained_mixt") or {})
    if int(native.get("version", 1)) != 1:
        raise ValueError("Unsupported pretrained_mixt version")
    native["version"] = 1
    if native.get("train_scope", "gate_only") not in {"gate_only", "gate_experts", "all"}:
        raise ValueError("pretrained_mixt.train_scope must be gate_only, gate_experts, or all")
    compressed = list(getattr(model.config, "mixt_compress_layers", []) or []) if hasattr(model, "config") else []
    # All native projections must be represented in the exported architecture.
    if compressed and set(compressed) != set(selected):
        raise ValueError("pretrained_mixt requires selecting all mixt_compress_layers; use layers_to_patch='mixt'")
    supplied = native.get("projections")
    actual = {}
    for idx in selected:
        mlp = layers[idx].mlp
        if all(getattr(getattr(mlp, p), "is_mixt_projector", False) for p in PROJECTIONS):
            actual[str(idx)] = {p: projection_spec(getattr(mlp, p)) for p in PROJECTIONS}
        elif supplied is None:
            raise ValueError(f"Layer {idx} has no native MixT weights or reconstruction specification")
    if supplied is not None and actual and any(supplied.get(k) != v for k, v in actual.items()):
        raise ValueError("Native MixT checkpoint structure differs from the saved import specification")
    native["projections"] = supplied if supplied is not None else actual
    if set(native["projections"]) != {str(i) for i in selected}:
        raise ValueError("pretrained_mixt projection specification does not match selected layers")
    if cfg.get("global_expert_enabled", False):
        raise ValueError("pretrained_mixt preserves the trained ensemble; disable global_expert_enabled")
    if cfg.get("drop_tokens", False) or float(cfg.get("jitter_eps", 0.0)) or float(cfg.get("gate_dropout", 0.0)):
        raise ValueError("pretrained_mixt requires drop_tokens=False, jitter_eps=0, gate_dropout=0")
    return native


class PretrainedMixTLinear(SparseMiXTLinear):
    """SparseMiXT with exact imported operators, dense/low-rank edges and alpha."""

    def __init__(self, spec, cfg, *, device, dtype, source=None):
        nn.Module.__init__(self)
        self.spec = copy.deepcopy(spec)
        self.in_features, self.out_features = int(spec["in_features"]), int(spec["out_features"])
        self.d = int(spec["d"])
        self.main_in_features = resolve_main_dim(self.in_features, spec["main_in_features"], self.d)
        self.main_out_features = resolve_main_dim(self.out_features, spec["main_out_features"], self.d)
        self.tail_in_features = self.in_features - self.main_in_features
        self.tail_out_features = self.out_features - self.main_out_features
        native = dict(cfg.get("pretrained_mixt") or {})
        self.router_input_policy = native.get("router_input_policy", "main")
        if self.router_input_policy not in {"main", "full_real"}:
            raise ValueError("pretrained_mixt.router_input_policy must be main or full_real")
        # Retain source precision at the import boundary. Mixed precision training
        # can explicitly cast the resulting model, independently of import.
        if source is not None:
            dtype = source.mixt00.bricks.dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.removeprefix("torch."))
        gate_cfg = dict(cfg)
        gate_cfg.update(E=len(spec["positions"]), d=self.d, k_in=spec["k_in"])
        gate_cfg.setdefault("gate_type", "softmax")
        gate_cfg.setdefault("topk", len(spec["positions"]))
        uniform_init = bool(native.get("uniform_init", True))
        if uniform_init and gate_cfg["gate_type"] != "softmax" and int(gate_cfg["topk"]) != gate_cfg["E"]:
            raise ValueError("Uniform pretrained MixT import requires softmax or topk equal to the imported expert count")
        gate_dim = self.main_in_features if self.router_input_policy == "main" else self.in_features
        contract = gate_config_from_mapping(gate_cfg, data_dim=gate_dim)
        self.core = GatedADTNLayer(
            dim_input=self.main_in_features, dim_output=self.main_out_features,
            num_blocks=gate_cfg["E"], d=self.d, k_in=spec["k_in"], gate_config=contract,
            pos_strategy="sliding", operator_positions=spec["positions"],
            native_mixt_accumulation=True, device=device, dtype=dtype)
        if self.core.k_out != int(spec["k_out"]):
            raise ValueError("Imported k_out disagrees with projection dimensions")
        if self.router_input_policy == "full_real":
            self.core.gate = gate_factory_config(contract)
        self.core.gate.router.register_parameter("identity_gain", nn.Parameter(torch.tensor(0.0 if uniform_init else 1.0)))
        self.core.gate.to(device=device, dtype=torch.float32)
        for block in self.core.blocks:
            block.register_parameter("operator_bias", nn.Parameter(torch.zeros((), device=device, dtype=dtype)))
        for name, incoming, outgoing in (
            ("01", self.tail_in_features, self.main_out_features),
            ("10", self.main_in_features, self.tail_out_features),
            ("11", self.tail_in_features, self.tail_out_features),
        ):
            branch_spec = spec["branches"][name]
            branch = None
            if incoming and outgoing:
                if branch_spec is None:
                    raise ValueError(f"Missing MixT branch {name}")
                if branch_spec["mode"] == "dense":
                    branch = nn.Linear(incoming, outgoing, bias=False, dtype=dtype, device=device)
                elif branch_spec["mode"] == "lowrank":
                    branch = LowRankLinear(incoming, outgoing, int(branch_spec["rank"]), dtype=dtype, device=device)
                else:
                    raise ValueError(f"Unsupported MixT branch mode: {branch_spec}")
            elif branch_spec is not None:
                raise ValueError(f"Nonempty MixT branch {name} has empty dimensions")
            setattr(self, "lr" + name, branch)
            self.register_parameter("alpha" + name, None if branch is None else nn.Parameter(torch.ones((), device=device, dtype=dtype)))
        self.bias = nn.Parameter(torch.zeros(self.out_features, device=device, dtype=dtype)) if spec["bias"] else None
        self.q_in, self.q_out = self.core.q_in, self.core.q_out
        self.in_pad, self.out_pad = self.in_features, self.out_features
        if source is not None:
            self.import_parameters(source)

    @torch.no_grad()
    def import_parameters(self, source):
        state = source.state_dict()
        consumed = set()

        def copy_to(target, key, index=None):
            value = state[key] if index is None else state[key][index]
            if target.numel() != value.numel():
                raise ValueError(f"MixT tensor size mismatch: {key}")
            target.copy_(value.reshape_as(target).to(device=target.device, dtype=target.dtype))
            consumed.add(key)

        for idx, block in enumerate(self.core.blocks):
            copy_to(block.U, "mixt00.bricks", idx)
            copy_to(block.operator_bias, "mixt00.bias", idx)
        for target, original in (("01", "10"), ("10", "01"), ("11", "11")):
            branch = getattr(self, "lr" + target)
            if branch is None:
                continue
            if isinstance(branch, LowRankLinear):
                copy_to(branch.B.weight, "w" + original + ".0.weight")
                copy_to(branch.A.weight, "w" + original + ".1.weight")
            else:
                copy_to(branch.weight, "w" + original + ".weight")
            copy_to(getattr(self, "alpha" + target), "alpha" + original)
        if self.bias is not None:
            copy_to(self.bias, "bias")
        if consumed != set(state):
            raise ValueError(f"Unmapped native MixT parameters: {sorted(set(state) - consumed)}")


def resolve_mixt_layers(model, selection):
    if str(selection).strip().lower() != "mixt":
        from .patching import resolve_layer_idxs
        return resolve_layer_idxs(int(model.config.num_hidden_layers), selection)
    layers = list(getattr(model.config, "mixt_compress_layers", []) or [])
    if not layers:
        raise ValueError("layers_to_patch='mixt' requires nonempty mixt_compress_layers")
    return [int(i) for i in layers]
