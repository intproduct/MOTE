"""Deferred, token-weighted loss diagnostics without Trainer dependencies."""
from __future__ import annotations

from typing import Any, Dict

import torch as tc

DEFERRED_LOSS_METRICS_MARKER = "_deferred_loss_observability"
PENDING_LOSS_METRICS_KEY = "_pending_loss_observability"


def _merge_scalar_maps(target: Dict[str, tc.Tensor], source: Dict[str, tc.Tensor]) -> None:
    for key, value in source.items():
        value = value.detach()
        target[key] = value if key not in target else target[key] + value

def accumulate_loss_observability(runtime: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Accumulate detached loss diagnostics on-device until an output boundary."""
    if not metrics or not metrics.get(DEFERRED_LOSS_METRICS_MARKER):
        return
    pending = runtime.get(PENDING_LOSS_METRICS_KEY)
    if pending is None:
        pending = {
            DEFERRED_LOSS_METRICS_MARKER: True,
            "bucket_loss_sums": {},
            "bucket_token_counts": {},
            "task_loss_sums": {},
            "task_token_counts": {},
            "final_answer_loss_sum": metrics["final_answer_loss_sum"].detach(),
            "final_answer_weighted_tokens": metrics["final_answer_weighted_tokens"].detach(),
        }
        runtime[PENDING_LOSS_METRICS_KEY] = pending
    else:
        pending["final_answer_loss_sum"] = (
            pending["final_answer_loss_sum"] + metrics["final_answer_loss_sum"].detach()
        )
        pending["final_answer_weighted_tokens"] = (
            pending["final_answer_weighted_tokens"] + metrics["final_answer_weighted_tokens"].detach()
        )
    for name in ("bucket_loss_sums", "bucket_token_counts", "task_loss_sums", "task_token_counts"):
        _merge_scalar_maps(pending[name], metrics[name])

def materialize_loss_observability(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all deferred scalar diagnostics with a single device-to-host copy."""
    if not metrics or not metrics.get(DEFERRED_LOSS_METRICS_MARKER):
        return metrics or {}

    bucket_names = list(metrics["bucket_loss_sums"])
    task_names = list(metrics["task_loss_sums"])
    scalars = []
    for name in bucket_names:
        scalars.extend((metrics["bucket_loss_sums"][name], metrics["bucket_token_counts"][name]))
    for name in task_names:
        scalars.extend((metrics["task_loss_sums"][name], metrics["task_token_counts"][name]))
    scalars.extend((metrics["final_answer_loss_sum"], metrics["final_answer_weighted_tokens"]))
    host_values = tc.stack([value.detach().to(dtype=tc.float64) for value in scalars]).cpu().tolist()

    result: Dict[str, Any] = {}
    offset = 0
    for name in bucket_names:
        loss_sum, token_count = host_values[offset : offset + 2]
        offset += 2
        count = int(token_count)
        result[f"loss/{name}"] = None if count <= 0 else float(loss_sum) / count
        result[f"tokens/{name}"] = count
    for name in task_names:
        loss_sum, token_count = host_values[offset : offset + 2]
        offset += 2
        count = int(token_count)
        if count > 0:
            result[f"loss_by_task/{name}"] = float(loss_sum) / count
    final_loss_sum, final_token_count = host_values[offset : offset + 2]
    final_count = int(final_token_count)
    result["final_answer_weighted_tokens"] = final_count
    result["final_answer_loss"] = None if final_count <= 0 else float(final_loss_sum) / final_count
    return result

def flush_loss_observability(runtime: Dict[str, Any]) -> Dict[str, Any]:
    """Publish pending GPU diagnostics at a logging or checkpoint boundary."""
    pending = runtime.pop(PENDING_LOSS_METRICS_KEY, None)
    if pending is None:
        return runtime.get("loss_observability") or {}
    materialized = materialize_loss_observability(pending)
    runtime["loss_observability"] = materialized
    runtime["final_answer_weighted_tokens"] = materialized.get("final_answer_weighted_tokens", 0)
    runtime["final_answer_loss"] = materialized.get("final_answer_loss")
    return materialized
