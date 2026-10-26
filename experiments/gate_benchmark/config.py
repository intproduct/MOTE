from __future__ import annotations
import hashlib
import json
from copy import deepcopy
from pathlib import Path

GATES = ("mlp", "compact_born", "mps")
IMAGE_TASKS = ("mnist", "fashion_mnist", "cifar10")
TEXT_TASKS = ("sst2", "mrpc", "rte")
DEFAULT = {
    "cache_dir": "outputs/gate_cache",
    "output_dir": "outputs/gate_benchmark",
    "device": "cuda",
    "precision": "fp32",
    "seeds": [17, 29, 43],
    "gates": list(GATES),
    "data_seed": 20261027,
    "deterministic": True,
    "experts": 16,
    "topk": 4,
    "k_in": 4,
    "drop_tokens": False,
    "aux_coeff": 0.01,
    "temperature": 1.0,
    "norm": "layernorm",
    "budget": "matched",
    "rank": 16,
    "sites": 16,
    "bond_dim": 8,
    "depth": 2,
    "quantum_entangle": True,
    "learnable_gain": True,
    "output_init_std": 0.001,
    "bert_model": "google/bert_uncased_L-4_H-256_A-4",
    "bert_revision": "main",
    "bert_layers": 3,
    "max_length": 128,
    "train_limit": 10000,
    "dev_limit": 1000,
    "test_limit": 2000,
    "dev_fraction": 0.15,
    "eval_every": 100,
    "log_every": 25,
    "mlp": {
        "tasks": list(IMAGE_TASKS),
        "width": 1024,
        "layers": 4,
        "expansion": 2,
        "batch_size": 128,
        "steps": 1000,
        "lr": 0.001,
        "weight_decay": 0.01,
        "warm_init_steps": 0,
    },
    "bert": {
        "tasks": list(TEXT_TASKS),
        "batch_size": 16,
        "steps": 600,
        "lr": 0.0001,
        "weight_decay": 0.01,
        "warm_init_steps": 40,
    },
}


def merged(base, update):
    out = deepcopy(base)
    for k, v in update.items():
        out[k] = (
            merged(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else deepcopy(v)
        )
    return out


def load_config(path=None):
    cfg = merged(DEFAULT, json.loads(Path(path).read_text()) if path else {})
    validate(cfg)
    return cfg


def validate(c):
    e = c["experts"]
    if e < 2 or e > 64 or e & (e - 1):
        raise ValueError("experts must be one of 2,4,8,16,32,64")
    if not 1 <= c["topk"] <= e:
        raise ValueError("topk must be in [1,experts]")
    if not c["gates"] or not set(c["gates"]) <= set(GATES):
        raise ValueError(f"gates must be selected from {GATES}")
    if c["precision"] not in {"fp32", "fp16"} or c["budget"] not in {"matched", "rank"}:
        raise ValueError("precision=fp32/fp16 and budget=matched/rank required")
    if c["bert_layers"] not in {2, 3}:
        raise ValueError("bert_layers must be 2 or 3")
    if not 0 < c["dev_fraction"] < 0.5:
        raise ValueError("dev_fraction must be in (0,0.5)")
    if c["depth"] not in {1, 2, 3, 4}:
        raise ValueError("depth must be 1..4")
    for key in (
        "rank",
        "sites",
        "bond_dim",
        "train_limit",
        "dev_limit",
        "test_limit",
        "eval_every",
        "log_every",
        "max_length",
    ):
        if c[key] < 1:
            raise ValueError(f"{key} must be positive")
    for stage, allowed in (("mlp", IMAGE_TASKS), ("bert", TEXT_TASKS)):
        if not set(c[stage]["tasks"]) <= set(allowed):
            raise ValueError(f"invalid tasks for {stage}")
        for k in ("batch_size", "steps", "lr"):
            if c[stage][k] <= 0:
                raise ValueError(f"{stage}.{k} must be positive")
    if not c["seeds"] or len(c["seeds"]) != len(set(c["seeds"])):
        raise ValueError("provide distinct seeds")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def data_key(cfg):
    return digest(
        {
            k: cfg[k]
            for k in (
                "data_seed",
                "train_limit",
                "dev_limit",
                "test_limit",
                "dev_fraction",
                "max_length",
                "bert_model",
                "bert_revision",
            )
        }
    )[:16]
