from __future__ import annotations

from typing import Dict, List

import torch as tc


def pad_collate(
    batch: List[Dict[str, tc.Tensor]],
    pad_id: int,
    *,
    pad_to_length_buckets: List[int] | None = None,
) -> Dict[str, tc.Tensor]:
    observed_max_len = max(x["input_ids"].numel() for x in batch)
    buckets = sorted(set(int(value) for value in (pad_to_length_buckets or [])))
    if buckets:
        candidates = [value for value in buckets if value >= observed_max_len]
        if not candidates:
            raise ValueError(
                f"batch length {observed_max_len} exceeds largest configured padding bucket {buckets[-1]}"
            )
        max_len = candidates[0]
    else:
        max_len = observed_max_len
    input_ids = tc.full((len(batch), max_len), pad_id, dtype=tc.long)
    labels = tc.full((len(batch), max_len), -100, dtype=tc.long)
    attn = tc.zeros((len(batch), max_len), dtype=tc.long)
    has_loss_weights = any("loss_weights" in x for x in batch)
    loss_weights = tc.zeros((len(batch), max_len), dtype=tc.float32) if has_loss_weights else None
    task = [x.get("task", "unknown") for x in batch]
    group = [x.get("group", "unknown") for x in batch]
    bucket = [x.get("bucket", "unknown") for x in batch]
    source_family = [x.get("source_family", "unknown") for x in batch]
    eval_type = [x.get("eval_type", "unknown") for x in batch]
    sample_id = [x.get("sample_id", "") for x in batch]
    data_diagnostics = [dict(x.get("data_diagnostics") or {}) for x in batch]

    for i, ex in enumerate(batch):
        length = ex["input_ids"].numel()
        input_ids[i, :length] = ex["input_ids"]
        labels[i, :length] = ex["labels"]
        if "attention_mask" in ex:
            example_attention = ex["attention_mask"].to(dtype=tc.long)
            if example_attention.numel() != length:
                raise ValueError("attention_mask length must match input_ids length")
            attn[i, :length] = example_attention
        else:
            attn[i, :length] = 1
        if loss_weights is not None:
            if "loss_weights" in ex:
                loss_weights[i, :length] = ex["loss_weights"]
            else:
                loss_weights[i, :length] = (ex["labels"] != -100).to(tc.float32)

    result = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
        "task": task,
        "group": group,
        "bucket": bucket,
        "source_family": source_family,
        "eval_type": eval_type,
        "sample_id": sample_id,
        "data_diagnostics": data_diagnostics,
    }
    if loss_weights is not None:
        result["loss_weights"] = loss_weights
    return result
