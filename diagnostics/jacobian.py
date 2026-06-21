from __future__ import annotations

from typing import Any, Callable, Dict

import torch as tc


def _unwrap_y(out):
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def estimate_jvp_stats(
    module_fn: Callable[[tc.Tensor], tc.Tensor],
    x_sample: tc.Tensor,
    samples: int,
    *,
    estimate_top: bool = False,
) -> Dict[str, Any]:
    samples = int(samples)
    if samples <= 0:
        return {}
    x = x_sample.detach().clone().requires_grad_(True)
    norms = []
    sq = []
    with tc.enable_grad():
        for _ in range(samples):
            v = tc.randn_like(x)
            y, jv = tc.autograd.functional.jvp(lambda z: _unwrap_y(module_fn(z)), (x,), (v,), create_graph=False, strict=False)
            n = jv.detach().float().norm()
            norms.append(n)
            sq.append(n.pow(2))
    vals = tc.stack(norms)
    sq_vals = tc.stack(sq)
    in_dim = int(x[0].numel()) if x.ndim > 1 and x.shape[0] > 0 else int(x.numel())
    result = {
        "jvp_norm_mean": float(vals.mean().item()),
        "jvp_norm_std": float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
        "fro_estimate": float((sq_vals.mean() * float(in_dim)).sqrt().item()),
        "trace_jtj_estimate": float((sq_vals.mean() * float(in_dim)).item()),
    }
    if estimate_top:
        result["top_singular_estimate_optional"] = float(vals.max().item())
    return result
