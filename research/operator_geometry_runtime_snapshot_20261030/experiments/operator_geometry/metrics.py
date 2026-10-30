"""Exact last-layer counterfactuals and local output-distribution geometry."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def substitution_indices(weights, k):
    """All k(E-k) replacements, ordered by selected and unselected score rank."""
    order = weights.argsort(-1, descending=True)
    selected, other = order[:, :k], order[:, k:]
    a = selected[:, :, None].expand(-1, k, weights.shape[-1]-k).flatten(1)
    b = other[:, None, :].expand(-1, k, -1).flatten(1)
    return a, b


def fixed_mass_delta(outputs, weights, source, target):
    """Move the selected expert's weight to one currently unselected expert."""
    n = torch.arange(len(outputs), device=outputs.device)
    return weights[n, source, None] * (outputs[n, target] - outputs[n, source])


def head(model, hidden):
    return F.linear(model.norm(hidden), model.embedding.weight)


def head_jvp(model, hidden, delta):
    """Analytic JVP of LayerNorm -> tied vocabulary head; bias has zero JVP."""
    centered = hidden - hidden.mean(-1, keepdim=True)
    dcentered = delta - delta.mean(-1, keepdim=True)
    inv = (centered.square().mean(-1, keepdim=True) + model.norm.eps).rsqrt()
    dnorm = dcentered * inv - centered * inv.pow(3) * (centered*dcentered).mean(-1, keepdim=True)
    return F.linear(dnorm * model.norm.weight, model.embedding.weight)


def compare_outputs(reference, changed, jvp, labels):
    """FP64 probability arithmetic; Fisher approximates KL, not signed NLL."""
    lp, lq = reference.double().log_softmax(-1), changed.double().log_softmax(-1)
    p = lp.exp()
    tangent = jvp.double()
    mean = (p*tangent).sum(-1)
    fisher = .5*((p*tangent.square()).sum(-1)-mean.square()).clamp_min(0)
    idx = labels[..., None]
    delta_nll = (lp-lq).gather(-1, idx).squeeze(-1)
    return {
        'delta_nll': delta_nll,
        'kl': (p*(lp-lq)).sum(-1).clamp_min(0),
        'fisher_kl': fisher,
        'first_order_nll': mean-tangent.gather(-1, idx).squeeze(-1),
        'argmax_changed': (reference.argmax(-1) != changed.argmax(-1)),
    }


def average_ranks(x):
    x = np.asarray(x)
    order = np.argsort(x, kind='stable')
    values = x[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1]
    ends = np.r_[starts[1:], len(x)]
    ranks = np.empty(len(x), dtype=float)
    for a,b in zip(starts,ends):
        ranks[order[a:b]] = (a+b-1)/2
    return ranks


def spearman(a,b):
    a,b = average_ranks(a), average_ranks(b)
    if not len(a) or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a,b)[0,1])


def bootstrap_window_mean(values, window_ids, seed, repeats=1000):
    """Windows, not correlated tokens/replacements, are resampling units."""
    groups = np.unique(window_ids)
    means = np.array([np.asarray(values)[window_ids == g].mean() for g in groups])
    rng = np.random.default_rng(seed)
    estimates = means[rng.integers(len(groups), size=(repeats,len(groups)))].mean(1)
    return [float(q) for q in np.quantile(estimates, [.025,.975])]
