from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from fitmotn.config.loader import load_config_from_json
from fitmotn.gate import GateConfig, Quantum_layer_Gate, RouterLogits, TopKGate, gate_factory, gate_factory_config
from fitmotn.patching import build_patch_model_config
from fitmotn.quantum_gate import QuantumGate, apply_cry, apply_ry
from fitmotn.sparse_mixt import SparseMiXTLinear


def _dense_reference(gate: QuantumGate, angles: np.ndarray) -> np.ndarray:
    """Independent NumPy full-register simulation, including cross-block state.

    Deliberately exponential and limited to tiny test registers. Uses explicit
    Kronecker operators, not the production block-simulator helpers.
    """
    width = gate.num_experts
    eye = np.eye(2)
    p0, p1 = np.diag([1, 0]), np.diag([0, 1])

    def kron_ops(ops):
        value = np.ones((1, 1))
        for op in ops:
            value = np.kron(value, op)
        return value

    def rotation(angle):
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        return np.array([[c, -s], [s, c]])

    def one(theta, target):
        return kron_ops([rotation(theta) if q == target else eye for q in range(width)])

    offsets = gate.rotation.detach().numpy()
    scales = gate.encoding_scale.detach().numpy()
    edges = gate.coupling.detach().numpy()
    axes = gate.readout_rotation.detach().numpy()
    state = np.zeros(2**width)
    state[0] = 1
    for layer in range(gate.depth):
        for q in range(width):
            state = one(angles[q] * scales[layer, q] + offsets[layer, q], q) @ state
        edge_index = 0
        for start in range(0, width, gate.block_size):
            for q in range(start, min(start + gate.block_size, width) - 1):
                control, target = (q, q + 1) if layer % 2 == 0 else (q + 1, q)
                inactive = kron_ops([p0 if i == control else eye for i in range(width)])
                active = kron_ops([p1 if i == control else rotation(edges[layer, edge_index]) if i == target else eye for i in range(width)])
                state = (inactive + active) @ state
                edge_index += 1
    for q in range(width):
        state = one(axes[q], q) @ state
    np.testing.assert_allclose(state @ state, 1, atol=1e-12)
    return np.array([state @ kron_ops([np.diag([1, -1]) if i == q else eye for i in range(width)]) @ state for q in range(width)])


@pytest.mark.parametrize("experts,block_size,depth", [(1, 1, 1), (3, 2, 2), (4, 4, 3), (5, 4, 2), (6, 2, 4)])
def test_local_simulator_matches_full_register_unitary(experts, block_size, depth):
    torch.manual_seed(11)
    gate = QuantumGate(7, experts, block_size=block_size, depth=depth).double()
    with torch.no_grad():
        for parameter in (gate.rotation, gate.coupling, gate.readout_rotation):
            parameter.uniform_(-1.4, 1.4)
    angles = torch.randn(2, experts, dtype=torch.float64)
    actual = gate.expectations_from_angles(angles).detach().numpy()
    expected = np.stack([_dense_reference(gate, item.numpy()) for item in angles])
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_rotations_preserve_norm_and_cry_can_create_entanglement():
    state = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
    state = apply_ry(state, torch.tensor(math.pi / 2), 0, 2)
    state = apply_cry(state, torch.tensor(math.pi), 0, 1, 2)
    torch.testing.assert_close(state.square().sum(), torch.tensor(1., dtype=torch.float64))
    # Bell state: both Schmidt singular values nonzero; entropy is one bit.
    singular_values = torch.linalg.svdvals(state.reshape(2, 2))
    torch.testing.assert_close(singular_values.square(), torch.full((2,), 0.5, dtype=torch.float64))


def test_zero_couplings_reduce_to_independent_ry_features():
    gate = QuantumGate(5, 5, block_size=4).double()
    with torch.no_grad():
        gate.coupling.zero_()
    angles = torch.randn(3, 5, dtype=torch.float64)
    total = (angles.unsqueeze(1) * gate.encoding_scale + gate.rotation).sum(1) + gate.readout_rotation
    torch.testing.assert_close(gate.expectations_from_angles(angles), total.cos())


def test_circuit_gradient_matches_finite_difference_for_inputs_and_parameters():
    gate = QuantumGate(5, 3, block_size=2).double()
    angles = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(gate.expectations_from_angles, (angles,), eps=1e-6, atol=1e-5)
    objective = lambda: (gate.expectations_from_angles(angles) * torch.tensor([1., -0.8, 0.4])).sum()
    objective().backward()
    for parameter in (gate.rotation, gate.encoding_scale, gate.coupling, gate.readout_rotation):
        analytical = parameter.grad.flatten()[0].item()
        original = parameter.detach().flatten()[0].item()
        with torch.no_grad():
            parameter.flatten()[0] = original + 1e-6
            plus = objective().item()
            parameter.flatten()[0] = original - 1e-6
            minus = objective().item()
            parameter.flatten()[0] = original
        assert analytical == pytest.approx((plus - minus) / 2e-6, abs=2e-6)


@pytest.mark.parametrize("readout", ["born", "expectation"])
def test_router_contract_backward_and_state_dict_roundtrip(readout):
    torch.manual_seed(7)
    config = GateConfig(data_dim=7, num_experts=5, k=2, drop_tokens=False,
                        gate_arch="quantum", gate_quantum={"rank": 3, "readout": readout})
    gate = gate_factory_config(config)
    x = torch.randn(8, 7, requires_grad=True)
    probs, mask, aux = gate(x)
    torch.testing.assert_close(probs.sum(-1), torch.ones(8))
    assert (mask.sum(-1) == 2).all()
    (probs * torch.arange(5)).sum().add(aux["l_aux"]).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in gate.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        if parameter.numel():
            assert parameter.grad.abs().sum() > 0, name
    restored = gate_factory_config(config)
    restored.load_state_dict(gate.state_dict(), strict=True)
    torch.testing.assert_close(restored(x.detach())[0], probs.detach())
    assert gate.collect_runtime_usage_tensors()["expert_counts"].sum() == 16


def test_born_readout_matches_normalized_marginals_and_temperature():
    gate = gate_factory_config(GateConfig(gate_type="softmax", data_dim=6, num_experts=5,
                                        temperature=0.7, gate_arch="quantum"))
    x = torch.randn(4, 6)
    quantum = gate.router.quantum
    marginal = ((1 - quantum.expectations(x)) / 2).clamp(0, 1)
    marginal = quantum.probability_floor + (1 - 2 * quantum.probability_floor) * marginal
    powered = marginal.pow(1 / 0.7)
    torch.testing.assert_close(gate(x)[0], powered / powered.sum(-1, keepdim=True))


def test_negative_logits_do_not_lose_capacity_to_unselected_routes():
    gate = TopKGate(2, 2, k=1, min_capacity=1, capacity_factor=1, drop_tokens=True)
    logits = torch.tensor([[-1., -8.], [-2., -9.], [-8., -1.], [-9., -2.]])
    selected = logits.argmax(-1, keepdim=True)
    mask, _ = gate._apply_capacity(logits, selected, cap=2)
    torch.testing.assert_close(mask, torch.tensor([[True, False], [True, False], [False, True], [False, True]]))


def test_quantum_uses_identical_capacity_and_aux_formulas():
    gate = gate_factory_config(GateConfig(data_dim=6, num_experts=4, k=2, min_capacity=1,
                                        capacity_factor=0.5, gate_arch="quantum", zloss_coeff=0.02))
    x = torch.randn(12, 6)
    logits = gate.router(x)

    class FixedLogits(nn.Module):
        def forward(self, unused):
            return logits

    control = TopKGate(6, 4, k=2, min_capacity=1, capacity_factor=0.5, zloss_coeff=0.02)
    control.router = FixedLogits()
    a, ma, aa = gate(x)
    b, mb, ab = control(x)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(ma, mb)
    for key in aa:
        torch.testing.assert_close(aa[key], ab[key])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cast_router_keeps_circuit_finite_and_differentiable(dtype):
    router = RouterLogits(9, 5, gate_arch="quantum").to(dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = router(torch.randn(2, 3, 9))
    assert logits.shape == (2, 3, 5) and logits.dtype == torch.float32
    logits.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in router.parameters() if p.numel())


def test_config_json_and_sparse_mixt_training_integration(tmp_path):
    path = tmp_path / "quantum.json"
    path.write_text(json.dumps({"runtime": {"dev_mode": True}, "model": {
        "model_path": str(tmp_path / "model"), "gate_arch": "quantum",
        "gate_quantum": {"rank": 3, "depth": 2, "block_size": 2},
    }}))
    cfg = load_config_from_json(path)
    patch = build_patch_model_config(cfg)
    assert patch["gate_quantum"]["rank"] == 3
    patch.update(E=2, d=2, k_in=1, topk=2, drop_tokens=False, pos_strategy="sliding")
    sparse = SparseMiXTLinear(6, 10, cfg=patch, ranks={"01": 2, "10": 2, "11": 2}, main_in_features=4, main_out_features=8)
    y, _ = sparse(torch.randn(5, 6))
    y.square().mean().backward()
    quantum = sparse.core.gate.router.quantum
    assert quantum.num_experts == sparse.core.num_blocks
    assert quantum.angle_proj.weight.grad.abs().sum() > 0


def test_bounded_size_and_legacy_entrypoint():
    gate = gate_factory("quantum", 64, 65, k=3, drop_tokens=False, gate_quantum={"rank": 8})
    assert sum(p.numel() for p in gate.parameters()) < 2000
    probs, mask, _ = gate(torch.randn(3, 64))
    assert probs.shape == (3, 65) and (mask.sum(-1) == 3).all()
    legacy = Quantum_layer_Gate(6, 3, dims=2, k=2, drop_tokens=False)
    probs, mask, aux = legacy(torch.randn(2, 4, 6))
    assert probs.shape == mask.shape == (2, 4, 3) and "l_aux" in aux


@pytest.mark.parametrize("kwargs", [{"depth": 5}, {"block_size": 8}, {"rank": 0}, {"readout": "bad"}, {"probability_floor": 0}])
def test_invalid_complexity_or_readout_rejected(kwargs):
    with pytest.raises(ValueError):
        QuantumGate(8, 4, **kwargs)


@pytest.mark.parametrize("backend", ["motn", "sparse_mixt"])
def test_quantum_hf_model_save_reload_and_backward(tmp_path, backend):
    from transformers import Qwen2Config
    from fitmotn.export.remote_code.configuration_fitmotn import FitMoTNConfig
    from fitmotn.export.remote_code.modeling_fitmotn import FitMoTNForCausalLM

    base = Qwen2Config(vocab_size=24, hidden_size=8, intermediate_size=16,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=32, tie_word_embeddings=False)
    patch = {"patch_backend": backend, "gate_type": "topk", "gate_arch": "quantum",
             "gate_quantum": {"rank": 3, "depth": 2, "block_size": 2},
             "E": 2, "topk": 1, "d": 2, "k_in": 2, "drop_tokens": False,
             "pos_strategy": "sliding", "dtype": "float32"}
    if backend == "sparse_mixt":
        patch["sparse_mixt"] = {"hidden_main": 8, "intermediate_main": 16,
                                "router_input_policy": "main"}
    config = FitMoTNConfig(base_model_name_or_path="tiny-local",
                          base_model_config=base.to_dict(), fitmotn_patch_config=patch,
                          layers_to_patch=[0], patch_backend=backend)
    model = FitMoTNForCausalLM(config).eval()
    ids = torch.tensor([[1, 3, 7, 5]])
    before = model(input_ids=ids, use_cache=False).logits
    before.square().mean().backward()
    routers = [m for m in model.modules() if isinstance(m, QuantumGate)]
    assert len(routers) == 3
    assert all(m.angle_proj.weight.grad is not None for m in routers)
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = FitMoTNForCausalLM.from_pretrained(tmp_path).eval()
    assert restored.config.fitmotn_patch_config["gate_quantum"]["rank"] == 3
    with torch.no_grad():
        after = restored(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(before.detach(), after, atol=1e-6, rtol=1e-5)
    restored.to(torch.bfloat16)
    with torch.no_grad():
        assert torch.isfinite(restored(input_ids=ids, use_cache=False).logits).all()


def test_empty_token_batch_has_valid_shape():
    gate = QuantumGate(8, 5)
    assert gate(torch.empty(0, 8)).shape == (0, 5)
