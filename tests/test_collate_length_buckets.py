from __future__ import annotations

import pytest
import torch

from fitmotn.data.collate import pad_collate


def _example(length: int, *, weighted: bool = False):
    value = {
        "input_ids": torch.arange(length, dtype=torch.long),
        "labels": torch.arange(length, dtype=torch.long),
    }
    if weighted:
        value["loss_weights"] = torch.ones(length, dtype=torch.float32)
    return value


def test_pad_collate_uses_smallest_frozen_length_bucket():
    result = pad_collate(
        [_example(65, weighted=True), _example(127)],
        pad_id=99,
        pad_to_length_buckets=[128, 256, 512, 1024, 1280],
    )
    assert tuple(result["input_ids"].shape) == (2, 128)
    assert tuple(result["labels"].shape) == (2, 128)
    assert tuple(result["attention_mask"].shape) == (2, 128)
    assert tuple(result["loss_weights"].shape) == (2, 128)
    assert result["attention_mask"][0].sum().item() == 65
    assert result["labels"][0, 65:].eq(-100).all()
    assert result["loss_weights"][0, 65:].eq(0).all()


def test_pad_collate_preserves_legacy_dynamic_padding_without_buckets():
    result = pad_collate([_example(5), _example(9)], pad_id=0)
    assert tuple(result["input_ids"].shape) == (2, 9)


def test_pad_collate_fails_when_batch_exceeds_frozen_buckets():
    with pytest.raises(ValueError, match="exceeds largest configured padding bucket"):
        pad_collate([_example(129)], pad_id=0, pad_to_length_buckets=[64, 128])
