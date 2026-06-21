from __future__ import annotations

import math
from typing import Any, Dict

import torch as tc


def _matrix(x: tc.Tensor, max_tokens: int | None = None) -> tc.Tensor:
    if x.ndim < 2:
        x = x.reshape(-1, 1)
    else:
        x = x.reshape(-1, x.shape[-1])
    if max_tokens is not None and max_tokens > 0:
        x = x[: int(max_tokens)]
    return x.detach().to(device="cpu", dtype=tc.float32)


def _top_svd(centered: tc.Tensor, top_k: int) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor]:
    if centered.numel() == 0 or centered.shape[0] == 0 or centered.shape[1] == 0:
        return tc.empty(0), tc.empty(0), tc.empty((centered.shape[1], 0))
    k = max(1, min(int(top_k), min(centered.shape)))
    try:
        u, s, vh = tc.linalg.svd(centered, full_matrices=False)
        return s.contiguous(), s[:k].contiguous(), vh[:k].T.contiguous()
    except RuntimeError:
        u, s, v = tc.pca_lowrank(centered, q=k, center=False)
        return s.contiguous(), s[:k].contiguous(), v[:, :k].contiguous()


def _rank_metrics(s: tc.Tensor, s_top: tc.Tensor, denom: int) -> Dict[str, float]:
    if s.numel() == 0:
        return {"effective_rank": 0.0, "stable_rank": 0.0, "top1_energy_ratio": 0.0, "topk_energy_ratio": 0.0}
    eig = (s.float() ** 2) / float(max(1, denom))
    total = eig.sum().clamp_min(1e-30)
    p = (eig / total).clamp_min(1e-30)
    effective_rank = float(tc.exp(-(p * p.log()).sum()).item())
    stable_rank = float((s.pow(2).sum() / s.max().pow(2).clamp_min(1e-30)).item())
    eig_top = (s_top.float() ** 2) / float(max(1, denom))
    return {
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "top1_energy_ratio": float((eig[0] / total).item()),
        "topk_energy_ratio": float((eig_top.sum() / total).item()) if eig_top.numel() else 0.0,
    }


def spectral_stats(matrix: tc.Tensor, top_k_svd: int = 128, max_tokens: int | None = None) -> Dict[str, Any]:
    x = _matrix(matrix, max_tokens=max_tokens)
    n, d = x.shape if x.ndim == 2 else (0, 0)
    if n == 0 or d == 0:
        return {
            "shape": [int(n), int(d)],
            "mean_norm": 0.0,
            "var_mean": 0.0,
            "var_std": 0.0,
            "covariance_trace": 0.0,
            "covariance_frobenius_norm": 0.0,
            "singular_values_topk": [],
            "covariance_eigenvalues_topk": [],
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "top1_energy_ratio": 0.0,
            "topk_energy_ratio": 0.0,
        }
    mean = x.mean(dim=0)
    centered = x - mean
    denom = max(1, n - 1)
    var = centered.pow(2).sum(dim=0) / float(denom)
    s_all, s, _ = _top_svd(centered, top_k_svd)
    eig_all = (s_all.float() ** 2) / float(denom)
    eig = (s.float() ** 2) / float(denom)
    cov_trace = float(var.sum().item())
    cov_fro = float(tc.sqrt(eig_all.pow(2).sum()).item()) if eig_all.numel() else 0.0
    metrics = _rank_metrics(s_all, s, denom)
    return {
        "shape": [int(n), int(d)],
        "mean_norm": float(mean.norm().item()),
        "var_mean": float(var.mean().item()) if var.numel() else 0.0,
        "var_std": float(var.std(unbiased=False).item()) if var.numel() else 0.0,
        "covariance_trace": cov_trace,
        "covariance_frobenius_norm": cov_fro,
        "singular_values_topk": [float(v) for v in s.tolist()],
        "covariance_eigenvalues_topk": [float(v) for v in eig.tolist()],
        **metrics,
    }


def _cosine_mean(a: tc.Tensor, b: tc.Tensor) -> float:
    denom = a.norm(dim=-1) * b.norm(dim=-1)
    mask = denom > 0
    if not bool(mask.any()):
        return 0.0
    return float(((a[mask] * b[mask]).sum(dim=-1) / denom[mask].clamp_min(1e-12)).mean().item())


def _principal_overlap(a: tc.Tensor, b: tc.Tensor, k: int) -> float:
    a = _matrix(a)
    b = _matrix(b)
    n = min(a.shape[0], b.shape[0])
    d = min(a.shape[1], b.shape[1])
    if n == 0 or d == 0:
        return 0.0
    a = a[:n, :d] - a[:n, :d].mean(dim=0)
    b = b[:n, :d] - b[:n, :d].mean(dim=0)
    kk = max(1, min(int(k), min(a.shape), min(b.shape)))
    _, _, va = _top_svd(a, kk)
    _, _, vb = _top_svd(b, kk)
    if va.numel() == 0 or vb.numel() == 0:
        return 0.0
    return float((tc.linalg.norm(va.T @ vb, ord="fro").pow(2) / float(kk)).item())


def compare_matrices(a: tc.Tensor, b: tc.Tensor, top_k_svd: int = 128, max_tokens: int | None = None) -> Dict[str, Any]:
    x = _matrix(a, max_tokens=max_tokens)
    y = _matrix(b, max_tokens=max_tokens)
    n = min(x.shape[0], y.shape[0])
    d = min(x.shape[1], y.shape[1])
    x = x[:n, :d]
    y = y[:n, :d]
    if n == 0 or d == 0:
        return {}
    diff = y - x
    xs = spectral_stats(x, top_k_svd=top_k_svd)
    ys = spectral_stats(y, top_k_svd=top_k_svd)
    l2 = diff.norm(dim=-1)
    base = x.norm(dim=-1).mean().clamp_min(1e-12)
    var_x = x.var(dim=0, unbiased=False)
    var_y = y.var(dim=0, unbiased=False)
    sx = tc.tensor(xs["singular_values_topk"], dtype=tc.float32)
    sy = tc.tensor(ys["singular_values_topk"], dtype=tc.float32)
    k = min(sx.numel(), sy.numel())
    slog = 0.0 if k == 0 else float((tc.log1p(sx[:k]) - tc.log1p(sy[:k])).norm().item())
    eig_x = tc.tensor(xs["covariance_eigenvalues_topk"], dtype=tc.float32)
    eig_y = tc.tensor(ys["covariance_eigenvalues_topk"], dtype=tc.float32)
    kk = min(eig_x.numel(), eig_y.numel())
    cov_fro_diff = 0.0 if kk == 0 else float((eig_x[:kk] - eig_y[:kk]).norm().item())
    return {
        "pointwise_l2_mean": float(l2.mean().item()),
        "pointwise_l2_relative": float((l2.mean() / base).item()),
        "cosine_mean": _cosine_mean(x, y),
        "mean_diff_norm": float((y.mean(dim=0) - x.mean(dim=0)).norm().item()),
        "var_diff_norm": float((var_y - var_x).norm().item()),
        "cov_fro_diff": cov_fro_diff,
        "singular_value_log_l2": slog,
        "log_spectrum_distance": slog,
        "covariance_fro_distance": cov_fro_diff,
        "effective_rank_dense": xs["effective_rank"],
        "effective_rank_mote": ys["effective_rank"],
        "effective_rank_ratio": float(ys["effective_rank"] / max(xs["effective_rank"], 1e-12)),
        "stable_rank_dense": xs["stable_rank"],
        "stable_rank_mote": ys["stable_rank"],
        "principal_subspace_overlap@k": _principal_overlap(x, y, min(top_k_svd, d, n)),
    }
