"""Full replay state, FP32-master AdamW, and token-based learning rate schedule."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def fresh_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


class MasterAdamW:
    """Explicit FP32 masters and moments, even when expert parameters are BF16/FP16."""
    def __init__(self, model, expert_lr=3e-5, router_lr=1e-4, weight_decay=0.01):
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.names = [n for n, _ in named]
        self.parameters = [p for _, p in named]
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters]
        groups = {}
        for (name, parameter), master in zip(named, self.masters):
            lr = router_lr if ".router." in name else expert_lr
            decay = 0.0 if parameter.ndim < 2 or name.endswith("bias") or ".norm." in name else weight_decay
            groups.setdefault((lr, decay), []).append(master)
        self.optimizer = torch.optim.AdamW([
            {"params": parameters, "lr": lr, "initial_lr": lr, "weight_decay": decay}
            for (lr, decay), parameters in groups.items()
        ], foreach=False)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.parameters:
            p.grad = None

    def step(self, scaler=None, max_grad_norm=1.0, capture_stats=False):
        for p, master in zip(self.parameters, self.masters):
            master.grad = None if p.grad is None else p.grad.detach().float().clone()
        scaled = scaler is not None and scaler.is_enabled()
        scale_before = scaler.get_scale() if scaled else 1.0
        if scaled:
            scaler.unscale_(self.optimizer)
        groups = {}
        if capture_stats:
            for name, master in zip(self.names, self.masters):
                group = "router" if ".router." in name else "experts" if ".experts." in name else "residual_tail_or_dense"
                groups.setdefault(group, []).append(master.grad.detach().norm() if master.grad is not None else master.new_zeros(()))
            grad_groups = {name: float(torch.stack(norms).norm()) for name, norms in groups.items()}
        if scaler is not None and scaler.is_enabled():
            finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.masters)
            norm = torch.nn.utils.clip_grad_norm_(self.masters, max_grad_norm) if finite else torch.tensor(float("inf"))
            scaler.step(self.optimizer)
            scaler.update()
        else:
            norm = torch.nn.utils.clip_grad_norm_(self.masters, max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            finite = True
        deltas = {}
        with torch.no_grad():
            for name, p, master in zip(self.names, self.parameters, self.masters):
                if capture_stats:
                    group = "router" if ".router." in name else "experts" if ".experts." in name else "residual_tail_or_dense"
                    deltas.setdefault(group, []).append((master.to(p.dtype).float() - p.float()).norm())
                p.copy_(master)
        result = {"updated": finite, "grad_norm": float(norm), "scale_before": scale_before,
                  "scale_after": scaler.get_scale() if scaled else 1.0}
        if capture_stats:
            result["groups"] = {name: {"unscaled_grad_norm": grad_groups[name],
                "model_update_norm": float(torch.stack(values).norm()),
                "changed_parameter_tensors": int((torch.stack(values) > 0).sum())} for name, values in deltas.items()}
        return result

    def state_dict(self):
        return {"names": self.names, "masters": [p.detach().clone() for p in self.masters],
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        if state["names"] != self.names:
            raise ValueError("Checkpoint trainable parameter order differs")
        with torch.no_grad():
            for master, value in zip(self.masters, state["masters"]):
                master.copy_(value)
        self.optimizer.load_state_dict(state["optimizer"])
        with torch.no_grad():
            for p, master in zip(self.parameters, self.masters):
                p.copy_(master)


class TokenSchedule:
    def __init__(self, optimizer: MasterAdamW, target_tokens: int, warmup_fraction=0.05):
        if target_tokens <= 0 or not 0 < warmup_fraction < 1:
            raise ValueError("Invalid token schedule")
        self.optimizer = optimizer
        self.target_tokens, self.warmup_fraction = target_tokens, warmup_fraction
        self.tokens = 0

    def set_next_update(self, update_tokens):
        progress = min((self.tokens + update_tokens) / self.target_tokens, 1.0)
        factor = progress / self.warmup_fraction if progress < self.warmup_fraction else (
            0.5 * (1 + math.cos(math.pi * (progress - self.warmup_fraction) / (1 - self.warmup_fraction))))
        for group in self.optimizer.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * factor

    def advance(self, update_tokens):
        self.tokens += update_tokens
        if self.tokens > self.target_tokens:
            raise ValueError("Supervised token budget exceeded")

    def state_dict(self):
        return {"tokens": self.tokens, "target_tokens": self.target_tokens, "warmup_fraction": self.warmup_fraction}

    def load_state_dict(self, state):
        if (state["target_tokens"], state["warmup_fraction"]) != (self.target_tokens, self.warmup_fraction):
            raise ValueError("Cannot silently change a resumed token schedule")
        self.tokens = state["tokens"]


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, model, optimizer, schedule, scaler, cursor, identity, extra=None):
    """New directory, complete state, content hash, then final commit marker."""
    path = fresh_directory(path)
    state = {"schema": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "schedule": schedule.state_dict(), "scaler": None if scaler is None else scaler.state_dict(),
             "rng": rng_state(), "cursor": cursor, "identity": identity, "extra": extra}
    temporary = path / "training.pt.partial"
    with temporary.open("xb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    output = path / "training.pt"
    os.replace(temporary, output)
    write_json(path / "complete.json", {"schema": 1, "created_epoch": time.time(), "identity": identity,
                                      "tokens": schedule.tokens, "cursor": cursor,
                                      "files": {"training.pt": {"sha256": digest(output), "bytes": output.stat().st_size}}})


def load_checkpoint(path, model, optimizer, schedule, scaler, identity):
    path = Path(path)
    manifest = json.loads((path / "complete.json").read_text(encoding="utf8"))
    if manifest["identity"] != identity or digest(path / "training.pt") != manifest["files"]["training.pt"]["sha256"]:
        raise ValueError("Checkpoint identity or content hash mismatch")
    state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    if state["identity"] != identity:
        raise ValueError("Embedded checkpoint identity mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    schedule.load_state_dict(state["schedule"])
    if state["scaler"] is not None:
        if scaler is None:
            raise ValueError("Checkpoint requires a GradScaler")
        scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return state["cursor"], state["extra"]
