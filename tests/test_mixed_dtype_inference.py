from __future__ import annotations

import torch

from fitmotn.ADTN import apply_block_to_sites
from fitmotn.gate import RouterLogits


def test_router_aligns_activation_to_parameter_dtype():
    router = RouterLogits(data_dim=4, num_experts=2).to(dtype=torch.float64)
    output = router(torch.randn(3, 4, dtype=torch.float32))
    assert output.dtype == torch.float64
    assert output.shape == (3, 2)


def test_tensor_contraction_aligns_activation_to_block_dtype():
    x_sites = torch.randn(2, 2, 2, dtype=torch.float64)
    block = torch.randn(2, 2, dtype=torch.float32)
    output = apply_block_to_sites(
        x_sites,
        block,
        [1],
        d=2,
        q_in=2,
        k_in=1,
        k_out=1,
    )
    assert output.dtype == torch.float32
    assert output.shape == (2, 2, 2)
