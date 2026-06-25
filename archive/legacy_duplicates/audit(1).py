from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch as tc


def to_jsonable(obj: Any):
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tc.dtype):
        return str(obj)
    if isinstance(obj, tc.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    try:
        import numpy as np

        if isinstance(obj, np.dtype):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except Exception:
        pass
    return obj


def json_dump(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def jsonl_append(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(obj), ensure_ascii=False) + "\n")


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def git_commit() -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return res.stdout.strip() or None
    except Exception:
        return None


def build_environment_snapshot() -> Dict[str, Any]:
    cuda_available = bool(tc.cuda.is_available())
    device_name = None
    device_count = 0
    if cuda_available:
        try:
            device_count = int(tc.cuda.device_count())
            if device_count > 0:
                device_name = tc.cuda.get_device_name(0)
        except Exception:
            device_count = 0
            device_name = None
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "git_commit": git_commit(),
        "torch_version": package_version("torch"),
        "transformers_version": package_version("transformers"),
        "datasets_version": package_version("datasets"),
        "cuda_available": cuda_available,
        "cuda_device_name": device_name,
        "device_count": device_count,
    }


def cuda_snapshot(device: tc.device | str | None = None) -> Dict[str, Optional[float]]:
    if not tc.cuda.is_available():
        return {
            "cuda_mem_alloc_mb": None,
            "cuda_mem_reserved_mb": None,
            "cuda_mem_peak_alloc_mb": None,
            "cuda_mem_peak_reserved_mb": None,
        }
    dev = tc.device(device) if device is not None else tc.device("cuda")
    idx = dev.index if dev.index is not None else tc.cuda.current_device()
    try:
        alloc = float(tc.cuda.memory_allocated(idx) / (1024 ** 2))
        reserved = float(tc.cuda.memory_reserved(idx) / (1024 ** 2))
        peak_alloc = float(tc.cuda.max_memory_allocated(idx) / (1024 ** 2))
        peak_reserved = float(tc.cuda.max_memory_reserved(idx) / (1024 ** 2))
    except Exception:
        alloc = reserved = peak_alloc = peak_reserved = None
    return {
        "cuda_mem_alloc_mb": alloc,
        "cuda_mem_reserved_mb": reserved,
        "cuda_mem_peak_alloc_mb": peak_alloc,
        "cuda_mem_peak_reserved_mb": peak_reserved,
    }


def host_memory_snapshot() -> Dict[str, Optional[float]]:
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        rss_mb = float(proc.memory_info().rss / (1024 ** 2))
        return {"cpu_ram_used_mb": rss_mb, "host_ram_used_mb": float(psutil.virtual_memory().used / (1024 ** 2))}
    except Exception:
        return {"cpu_ram_used_mb": None, "host_ram_used_mb": None}


def parameter_snapshot(model: tc.nn.Module) -> Dict[str, Any]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    ratio = float(trainable / total) if total > 0 else None
    return {"trainable_params": trainable, "total_params": total, "trainable_ratio": ratio}


def module_norm(model: tc.nn.Module, trainable_only: bool = False) -> Optional[float]:
    total = 0.0
    found = False
    with tc.no_grad():
        for p in model.parameters():
            if trainable_only and not p.requires_grad:
                continue
            found = True
            total += float(p.detach().float().pow(2).sum().item())
    if not found:
        return None
    return float(math.sqrt(total))


def grad_norm(model: tc.nn.Module) -> Dict[str, Optional[float]]:
    total = 0.0
    nan_count = 0
    inf_count = 0
    found = False
    with tc.no_grad():
        for p in model.parameters():
            g = p.grad
            if g is None:
                continue
            found = True
            g = g.detach()
            nan_count += int(tc.isnan(g).sum().item())
            inf_count += int(tc.isinf(g).sum().item())
            total += float(g.float().pow(2).sum().item())
    if not found:
        return {"grad_norm": None, "num_nan_grads": None, "num_inf_grads": None}
    return {"grad_norm": float(math.sqrt(total)), "num_nan_grads": nan_count, "num_inf_grads": inf_count}


def tensor_distribution_stats(values: tc.Tensor) -> Dict[str, Any]:
    if values is None:
        return {
            "usage": None,
            "top1": None,
            "entropy": None,
            "load_balance": None,
            "active_expert_count": None,
            "max_expert_share": None,
            "expert_cv": None,
            "pos": None,
        }
    x = values.detach().float().flatten()
    if x.numel() == 0:
        return {
            "usage": None,
            "top1": None,
            "entropy": None,
            "load_balance": None,
            "active_expert_count": None,
            "max_expert_share": None,
            "expert_cv": None,
            "pos": None,
        }
    total = float(x.sum().item())
    probs = x / max(total, 1e-12)
    probs = tc.clamp(probs, 1e-12, 1.0)
    entropy = float((-probs * probs.log()).sum().item())
    denom = float(math.log(max(2, probs.numel())))
    load_balance = float(entropy / denom) if denom > 0 else None
    mean = float(x.mean().item())
    std = float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0
    top1 = int(tc.argmax(probs).item())
    active = int((x > 0).sum().item())
    max_share = float(probs.max().item())
    cv = float(std / max(mean, 1e-12))
    return {
        "usage": to_jsonable(x),
        "top1": top1,
        "entropy": entropy,
        "load_balance": load_balance,
        "active_expert_count": active,
        "max_expert_share": max_share,
        "expert_cv": cv,
        "pos": None,
    }


def derive_stage_step(global_step: int, stage_start: int) -> int:
    return max(0, int(global_step) - int(stage_start))

