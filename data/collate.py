from __future__ import annotations

from typing import Dict, List

import torch as tc


def pad_collate(batch: List[Dict[str, tc.Tensor]], pad_id: int) -> Dict[str, tc.Tensor]:
    max_len = max(x["input_ids"].numel() for x in batch)
    input_ids = tc.full((len(batch), max_len), pad_id, dtype=tc.long)
    labels = tc.full((len(batch), max_len), -100, dtype=tc.long)
    attn = tc.zeros((len(batch), max_len), dtype=tc.long)
    task = [x.get("task", "unknown") for x in batch]
    group = [x.get("group", "unknown") for x in batch]
    source_family = [x.get("source_family", "unknown") for x in batch]
    eval_type = [x.get("eval_type", "unknown") for x in batch]

    for i, ex in enumerate(batch):
        length = ex["input_ids"].numel()
        input_ids[i, :length] = ex["input_ids"]
        labels[i, :length] = ex["labels"]
        attn[i, :length] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
        "task": task,
        "group": group,
        "source_family": source_family,
        "eval_type": eval_type,
    }
