"""Numerical/gradient regression for contraction caching and accumulator reuse."""
import copy
import unittest

import torch

from fitmotn.ADTN import GatedADTNLayer, apply_block_to_sites, _apply_block_to_sites_reference
from fitmotn.gate import GateConfig


def reference_dispatch(core, x):
    sites, b, s = core.reshape_in(x.float())
    probs, mask, aux = core.gate(x.float())
    n = sites.shape[0]
    probs, mask = probs.reshape(n, core.num_blocks), mask.reshape(n, core.num_blocks)
    out = torch.zeros((n, *([core.d] * core.q_out)), device=x.device, dtype=probs.dtype)
    for i, block in enumerate(core.blocks):
        ids = torch.where(mask[:, i])[0]
        if not ids.numel():
            continue
        y = _apply_block_to_sites_reference(sites.index_select(0, ids), block.U, core.positions[i],
            d=core.d, q_in=core.q_in, k_in=core.k_in, k_out=core.k_out)
        weight = probs.index_select(0, ids)[:, i].view(-1, *([1] * (y.ndim - 1)))
        out = torch.index_add(out, 0, ids, y * weight)
    if core._should_use_global_expert(None):
        out = out + core.global_expert_weight * core._apply_global_expert(sites)
    return core.reshape_out(out, b, s), aux


class ContractionOptimizationTest(unittest.TestCase):
    def test_noncontiguous_sites_and_gradients(self):
        torch.manual_seed(87)
        for q, positions, kout in [(4, [0, 2], 3), (5, [4, 1], 1), (3, [0, 1, 2], 0)]:
            x = torch.randn(3, *([2] * q), dtype=torch.float64).transpose(1, 2).requires_grad_()
            u = torch.randn(*([2] * (len(positions) + kout)), dtype=torch.float64, requires_grad=True)
            kw = dict(d=2, q_in=q, k_in=len(positions), k_out=kout)
            expected = _apply_block_to_sites_reference(x, u, positions, **kw)
            actual = apply_block_to_sites(x, u, positions, **kw)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            upstream = torch.randn_like(actual)
            g0 = torch.autograd.grad(expected, (x, u), upstream, retain_graph=True)
            g1 = torch.autograd.grad(actual, (x, u), upstream)
            for a, b in zip(g0, g1):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_capacity_router_global_and_all_gradients(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            for drop in (True, False):
                torch.manual_seed(92)
                gate = GateConfig(data_dim=32, num_experts=6, k=2,
                    capacity_factor=0.5, min_capacity=1, drop_tokens=drop)
                core = GatedADTNLayer(dim_input=32, dim_output=64, num_blocks=6, d=2, k_in=2,
                    gate_config=gate, global_expert_enabled=True).to(device)
                ref = copy.deepcopy(core)
                x = torch.randn(21, 32, device=device, requires_grad=True)
                rx = x.detach().clone().requires_grad_()
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    y, aux = core(x, return_aux=True)
                    ry, raux = reference_dispatch(ref, rx)
                    loss = y.float().square().mean() + aux["l_aux"]
                    rloss = ry.float().square().mean() + raux["l_aux"]
                torch.testing.assert_close(y, ry, rtol=0, atol=0)
                loss.backward()
                rloss.backward()
                torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
                for (name, p), (rname, rp) in zip(core.named_parameters(), ref.named_parameters()):
                    self.assertEqual(name, rname)
                    self.assertEqual(p.grad is None, rp.grad is None)
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad, rp.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
