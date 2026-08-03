from __future__ import annotations

import copy
from pathlib import Path
import sys
import time
import unittest

import torch
import torch.nn as nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fitmotn.model import SparseMiXTFFNLayer
from fitmotn.patching import patch_qwen_ffn_layers


class _MediumQwenMLP(nn.Module):
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 2560, *, dtype=torch.float32):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _DecoderLayer(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp


class _Decoder(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderLayer(mlp)])


def _cfg(*, exact: bool, dtype: torch.dtype):
    return {
        "patch_backend": "sparse_mixt",
        "E": 1 if exact else 8,
        "d": 2,
        "k_in": 9 if exact else 6,
        "topk": 1 if exact else 2,
        "gate_type": "topk",
        "capacity_factor": 4.0,
        "min_capacity": 4,
        "drop_tokens": False,
        "pos_strategy": "random",
        "global_expert_enabled": not exact,
        "dtype": dtype,
        "sparse_mixt": {
            "backend_version": 1,
            "hidden_main": 512,
            "intermediate_main": 2048,
            "router_input_policy": "main",
            "boundary_init": "zero",
            "gate_ranks": {"01": 256 if exact else 64, "10": 512 if exact else 64, "11": 256 if exact else 64},
            "up_ranks": {"01": 256 if exact else 64, "10": 512 if exact else 64, "11": 256 if exact else 64},
            "down_ranks": {"01": 512 if exact else 64, "10": 256 if exact else 64, "11": 256 if exact else 64},
        },
    }


def _patch(source: _MediumQwenMLP, cfg: dict, device: torch.device, dtype: torch.dtype):
    decoder = _Decoder(source)
    patch_qwen_ffn_layers(decoder, [0], cfg, device, dtype=dtype)
    layer = decoder.layers[0].mlp
    assert isinstance(layer, SparseMiXTFFNLayer)
    return layer


def _print_runtime(label: str, *, start: float):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"[CUDA] {label}: seconds={time.perf_counter()-start:.3f} allocated_mib={allocated:.1f} peak_mib={peak:.1f}")


def _require_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")


def test_cuda_medium_exact_qwen_ffn_replacement_and_trace():
    """Full-rank boundaries + one full-site expert must reproduce a dense FFN."""
    _require_cuda()
    torch.manual_seed(20260803)
    torch.cuda.manual_seed_all(20260803)
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dense = _MediumQwenMLP().to(device)
    reference = copy.deepcopy(dense).eval()
    sparse = _patch(dense, _cfg(exact=True, dtype=torch.float32), device, torch.float32).eval()

    init_start = time.perf_counter()
    reports = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        reports[name] = getattr(sparse, name).initialize_from_dense_weight(
            getattr(reference, name).weight,
            initialize_main_exact=True,
            boundary_method="full_svd",
        )
    _print_runtime("exact initialization", start=init_start)
    print(f"[init] reports={reports}")

    trace_input = torch.randn(1, 2, 768, device=device)
    with torch.no_grad():
        traced = sparse.forward_with_trace(trace_input)
    assert traced.shape == (1, 2, 768)

    x = torch.randn(2, 64, 768, device=device)
    run_start = time.perf_counter()
    with torch.no_grad():
        expected = reference(x)
        actual = sparse(x)
    _print_runtime("exact forward", start=run_start)

    error = (actual - expected).float()
    rel_l2 = error.norm() / expected.float().norm().clamp_min(1e-12)
    max_abs = error.abs().max()
    print(f"[equivalence] rel_l2={rel_l2.item():.8e} max_abs={max_abs.item():.8e}")
    # Full-rank FP32 SVD is algebraically exact, but cuSOLVER reconstruction
    # accumulates about 1e-4 error here.  Norm and max-error bounds avoid an
    # unstable relative comparison at reference elements that are near zero.
    assert rel_l2.item() < 3e-4
    assert max_abs.item() < 2e-4


def test_cuda_medium_bf16_routed_training_gradients_and_restore():
    """Exercise routed experts, boundaries, BF16 backward, optimizer, and restore."""
    _require_cuda()
    torch.manual_seed(20260804)
    torch.cuda.manual_seed_all(20260804)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    source = _MediumQwenMLP(dtype=dtype).to(device)
    cfg = _cfg(exact=False, dtype=dtype)
    sparse = _patch(source, cfg, device, dtype).train()
    sparse.forward_with_trace(torch.randn(1, 2, 768, device=device, dtype=dtype))

    parameter_count = sum(p.numel() for p in sparse.parameters())
    print(f"[network] parameters={parameter_count:,} experts=8 topk=2 dtype={dtype}")
    optimizer = torch.optim.AdamW(sparse.parameters(), lr=2e-3)
    x = torch.randn(2, 128, 768, device=device, dtype=dtype)
    target = torch.randn(2, 128, 768, device=device, dtype=dtype)

    train_start = time.perf_counter()
    losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = sparse(x)
        loss = (output.float() - target.float()).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(sparse.parameters(), 1.0)
        assert torch.isfinite(grad_norm)
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"[train] step={step} loss={losses[-1]:.8f} grad_norm={float(grad_norm):.8f}")
    _print_runtime("two BF16 training steps", start=train_start)

    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(sparse, proj_name)
        assert proj.core.gate.router.linear.weight.grad is not None
        for branch_name in ("lr01", "lr10", "lr11"):
            branch = getattr(proj, branch_name)
            assert branch.A.weight.grad is not None and torch.isfinite(branch.A.weight.grad).all()
            assert branch.B.weight.grad is not None and torch.isfinite(branch.B.weight.grad).all()

    state = {name: tensor.detach().clone() for name, tensor in sparse.state_dict().items()}
    restored = _patch(_MediumQwenMLP(dtype=dtype).to(device), cfg, device, dtype).eval()
    restored.load_state_dict(state, strict=True)
    sparse.eval()
    probe = torch.randn(1, 16, 768, device=device, dtype=dtype)
    with torch.no_grad():
        before = sparse(probe)
        after = restored(probe)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    print(f"[restore] strict_state_dict=True exact_output=True tensors={len(state)}")


if __name__ == "__main__":
    print(f"[device] torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
    test_cuda_medium_exact_qwen_ffn_replacement_and_trace()
    test_cuda_medium_bf16_routed_training_gradients_and_restore()
    print("[result] all medium CUDA Sparse MiXT tests passed")
