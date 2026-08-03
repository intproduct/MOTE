from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
import time
import unittest

import torch
import torch.nn as nn
from safetensors import safe_open

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fitmotn.model import MixedMiXTFFNLayer


DEFAULT_MODEL_PATH = Path(r"D:\AI\Models\Qwen3.5-4B")


class _RealQwenMLP(nn.Module):
    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, device: torch.device):
        super().__init__()
        self.gate_proj = nn.Linear(2560, 9216, bias=False, device=device, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(2560, 9216, bias=False, device=device, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(9216, 2560, bias=False, device=device, dtype=torch.bfloat16)
        self.act_fn = nn.SiLU()
        with torch.no_grad():
            self.gate_proj.weight.copy_(gate)
            self.up_proj.weight.copy_(up)
            self.down_proj.weight.copy_(down)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _ShapeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)


class _ShapeOnlyQwenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = _ShapeLinear(2560, 9216)
        self.up_proj = _ShapeLinear(2560, 9216)
        self.down_proj = _ShapeLinear(9216, 2560)
        self.act_fn = nn.SiLU()


def _model_path() -> Path:
    return Path(os.environ.get("FITMOTN_QWEN35_4B_PATH", str(DEFAULT_MODEL_PATH))).resolve()


def _require_local_cuda_model() -> Path:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")
    root = _model_path()
    if not (root / "model.safetensors.index.json").exists():
        raise unittest.SkipTest(f"local Qwen3.5-4B weights not found at {root}")
    return root


def _load_layer0_weights(root: Path, device: torch.device):
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    prefix = "model.language_model.layers.0.mlp."
    tensors = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        name = prefix + projection + ".weight"
        shard = root / index["weight_map"][name]
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            tensors.append(handle.get_tensor(name).to(device=device, dtype=torch.bfloat16))
    return tensors


def _exact_cfg():
    return {
        "patch_backend": "mixed_mixt",
        "E": 1,
        "d": 2,
        "k_in": 8,
        "topk": 1,
        "gate_type": "topk",
        "capacity_factor": 100.0,
        "min_capacity": 1,
        "drop_tokens": False,
        "pos_strategy": "random",
        "global_expert_enabled": False,
        "dtype": torch.bfloat16,
        "mixed_mixt": {
            "backend_version": 1,
            "hidden_main": 2048,
            "intermediate_main": 8192,
            "router_input_policy": "full_real",
            "dense_init": "full_site_exact",
            "gate_bonds": {"m00": 11, "m01": 9, "m10": 11, "m11": 9},
            "up_bonds": {"m00": 11, "m01": 9, "m10": 11, "m11": 9},
            "down_bonds": {"m00": 13, "m01": 10, "m10": 13, "m11": 10},
        },
    }


def _structured_cfg():
    cfg = _exact_cfg()
    cfg.update(E=8, topk=2, global_expert_enabled=True)
    cfg["mixed_mixt"] = {
        **cfg["mixed_mixt"],
        "dense_init": "none",
        "gate_bonds": {"m00": 8, "m01": 5, "m10": 6, "m11": 5},
        "up_bonds": {"m00": 8, "m01": 5, "m10": 6, "m11": 5},
        "down_bonds": {"m00": 10, "m01": 6, "m10": 8, "m11": 6},
    }
    return cfg


def _runtime(label: str, start: float):
    torch.cuda.synchronize()
    print(
        f"[CUDA] {label}: seconds={time.perf_counter()-start:.3f} "
        f"allocated_mib={torch.cuda.memory_allocated()/1024**2:.1f} "
        f"peak_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}"
    )


def test_local_qwen35_layer0_full_site_mixed_mixt_matches_real_dense_ffn():
    root = _require_local_cuda_model()
    device = torch.device("cuda:0")
    torch.manual_seed(20260805)
    torch.cuda.manual_seed_all(20260805)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    weights = _load_layer0_weights(root, device)
    dense = _RealQwenMLP(*weights, device=device).eval()
    del weights
    build_start = time.perf_counter()
    mixed = MixedMiXTFFNLayer(dense, _exact_cfg(), device=device, dtype=torch.bfloat16).to(device).eval()
    _runtime("real layer exact construction", build_start)

    calls = {"gate": 0, "up": 0, "down": 0}
    handles = []
    for key, proj_name in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
        handles.append(getattr(mixed, proj_name).core.gate.register_forward_hook(lambda *args, key=key: calls.__setitem__(key, calls[key] + 1)))
    try:
        mixed.forward_with_trace(torch.randn(1, 1, 2560, device=device, dtype=torch.bfloat16))
        x = torch.randn(1, 8, 2560, device=device, dtype=torch.bfloat16)
        forward_start = time.perf_counter()
        with torch.no_grad():
            expected = dense(x)
            actual = mixed(x)
        _runtime("real layer exact forward", forward_start)
    finally:
        for handle in handles:
            handle.remove()

    error = (actual - expected).float()
    rel_l2 = error.norm() / expected.float().norm().clamp_min(1e-12)
    max_abs = error.abs().max()
    print(f"[real equivalence] rel_l2={rel_l2.item():.8e} max_abs={max_abs.item():.8e} gate_calls={calls}")
    assert rel_l2.item() < 1e-2
    assert max_abs.item() < 1e-2
    # One trace forward plus one measured forward: each projection routes once
    # per invocation, not once per quadrant.
    assert calls == {"gate": 2, "up": 2, "down": 2}


def test_local_qwen35_structured_mixed_mixt_bf16_training():
    _require_local_cuda_model()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(20260806)
    torch.cuda.manual_seed_all(20260806)

    mixed = MixedMiXTFFNLayer(_ShapeOnlyQwenMLP(), _structured_cfg(), device=device, dtype=dtype).to(device).train()
    mixed.forward_with_trace(torch.randn(1, 1, 2560, device=device, dtype=dtype))
    params = sum(parameter.numel() for parameter in mixed.parameters())
    print(f"[structured network] parameters={params:,} E=8 topk=2 global=True dtype={dtype}")
    optimizer = torch.optim.AdamW(mixed.parameters(), lr=1e-3)
    x = torch.randn(1, 8, 2560, device=device, dtype=dtype)
    target = torch.randn(1, 8, 2560, device=device, dtype=dtype)
    start = time.perf_counter()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = mixed(x)
        loss = (output.float() - target.float()).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(mixed.parameters(), 1.0)
        assert torch.isfinite(grad_norm)
        optimizer.step()
        print(
            f"[structured train] step={step} loss={float(loss.detach()):.8f} "
            f"grad_norm={float(grad_norm.detach()):.8f}"
        )
    _runtime("real-dimension structured training", start)

    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(mixed, proj_name)
        router_grad = proj.core.gate.router.linear.weight.grad
        assert router_grad is not None and torch.isfinite(router_grad).all()
        for quadrant in proj.core.quadrants.values():
            selected_grads = [block.U.grad for block in quadrant.blocks if block.U.grad is not None]
            assert selected_grads and all(torch.isfinite(grad).all() for grad in selected_grads)
            assert quadrant.global_block is not None
            assert quadrant.global_block.U.grad is not None and torch.isfinite(quadrant.global_block.U.grad).all()


if __name__ == "__main__":
    print(f"[device] torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
    print(f"[model] {_model_path()}")
    test_local_qwen35_layer0_full_site_mixed_mixt_matches_real_dense_ffn()
    test_local_qwen35_structured_mixed_mixt_bf16_training()
    print("[result] all local Qwen3.5-4B Mixed MiXT tests passed")
