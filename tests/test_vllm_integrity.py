from __future__ import annotations

import pytest
import torch

from fitmotn.rl.vllm_integrity import _evenly_spaced_indices, _sample_tensor


def test_evenly_spaced_indices_stay_in_bounds_for_large_model_tensor():
    # This is the observed Stage 5E frozen-parameter size. A float32 linspace
    # rounds its count - 1 endpoint up to count and causes CUDA index_select to
    # raise a device-side assertion.
    count = 544_997_376

    indices = _evenly_spaced_indices(count, 16)

    assert indices[0] == 0
    assert indices[-1] == count - 1
    assert indices == sorted(indices)
    assert all(0 <= index < count for index in indices)


def test_sample_tensor_uses_deterministic_integer_positions():
    tensor = torch.arange(10)

    assert list(_sample_tensor(tensor, 4)) == [0, 3, 6, 9]
    assert list(_sample_tensor(tensor, 1)) == [0]


@pytest.mark.parametrize(
    ("count", "take"),
    [(0, 1), (10, 0), (10, 11)],
)
def test_evenly_spaced_indices_reject_invalid_ranges(count: int, take: int):
    with pytest.raises(ValueError):
        _evenly_spaced_indices(count, take)
