from __future__ import annotations

from typing import Dict, List

import torch as tc


def pad_collate(batch: List[Dict[str, tc.Tensor]], pad_id: int) -> Dict[str, tc.Tensor]:
    max_len = max(x["input_ids"].numel() for x in batch)
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
