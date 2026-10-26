from __future__ import annotations
import numpy as np
import pytest
import torch
from fitmotn.gate import RouterLogits, gate_config_from_mapping, gate_factory_config
from fitmotn.structured_routers import CompactBornGate, ResidualMPSGate


@pytest.mark.parametrize("experts", [2, 4, 8, 16, 64])
def test_born_joint_probabilities_and_all_parameter_gradients(experts):
    torch.manual_seed(8)
    model = CompactBornGate(7, experts, rank=3).double()
    x = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    logits = model(x)
    assert logits.shape == (3, experts)
    torch.testing.assert_close(
        logits.exp().sum(-1), torch.ones(3, dtype=torch.float64), atol=2e-12, rtol=2e-12
    )
    (logits * torch.randn_like(logits)).sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in model.parameters():
        if p.numel():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert p.grad.abs().sum() > 0


def test_born_matches_independent_dense_numpy_circuit():
    torch.manual_seed(9)
    model = CompactBornGate(5, 8, rank=2, depth=3).double()
    for p in (model.rotation, model.coupling, model.readout_rotation):
        with torch.no_grad():
            p.uniform_(-1, 1)
    angles = torch.randn(2, 3, 3, 2, dtype=torch.float64, requires_grad=True)
    probs = model.probabilities_from_angles(angles)
    q = model.qubits

    def kron(ops):
        value = np.ones((1, 1), dtype=complex)
        for op in ops:
            value = np.kron(value, op)
        return value

    def ry(t):
        return np.array(
            [[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]]
        )

    eye = np.eye(2)
    expected = []
    for item in angles.detach().numpy():
        state = np.zeros(2**q, dtype=complex)
        state[0] = 1
        theta = item + model.rotation.detach().numpy()
        edge = 0
        for layer in range(model.depth):
            for wire in range(q):
                state = (
                    kron(
                        [
                            ry(theta[layer, wire, 0]) if j == wire else eye
                            for j in range(q)
                        ]
                    )
                    @ state
                )
                rz = np.diag(np.exp(np.array([-0.5j, 0.5j]) * theta[layer, wire, 1]))
                state = kron([rz if j == wire else eye for j in range(q)]) @ state
            for wire in range(layer % 2, q - 1, 2):
                c, t = (wire, wire + 1) if layer % 2 == 0 else (wire + 1, wire)
                for control, target in ((c, t), (t, c)):
                    inactive = kron(
                        [np.diag([1, 0]) if j == control else eye for j in range(q)]
                    )
                    active = kron(
                        [
                            np.diag([0, 1])
                            if j == control
                            else ry(model.coupling[edge].item())
                            if j == target
                            else eye
                            for j in range(q)
                        ]
                    )
                    state = (inactive + active) @ state
                    edge += 1
        for wire in range(q):
            state = (
                kron(
                    [
                        ry(model.readout_rotation[wire].item()) if j == wire else eye
                        for j in range(q)
                    ]
                )
                @ state
            )
        expected.append(np.abs(state) ** 2)
    np.testing.assert_allclose(probs.detach().numpy(), expected, atol=3e-12)
    assert torch.autograd.gradcheck(
        model.probabilities_from_angles, (angles,), eps=1e-6, atol=1e-5
    )


def test_mps_matches_explicit_matrix_product_direction():
    torch.manual_seed(2)
    model = ResidualMPSGate(5, 8, sites=4, bond_dim=3).double()
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    values = torch.tanh(model.input_proj(x))
    state = model.boundary.expand(2, 3)
    for i in range(4):
        matrix = (
            torch.eye(3, dtype=x.dtype) + values[:, i, None, None] * model.cores[i] / 2
        )
        state = torch.einsum("bi,bij->bj", state, matrix)
    state = torch.nn.functional.normalize(state, dim=-1)
    torch.testing.assert_close(model(x), model.output_proj(state))
    assert torch.autograd.gradcheck(model, (x,))


@pytest.mark.parametrize("arch", ["compact_born", "mps"])
def test_factory_aux_capacity_and_state_restore(arch):
    cfg = gate_config_from_mapping(
        dict(
            E=8,
            topk=2,
            gate_type="topk",
            gate_arch=arch,
            gate_learnable_gain=True,
            gate_mps=dict(sites=3, bond_dim=4),
            gate_quantum=dict(rank=3, depth=2),
        ),
        data_dim=6,
    )
    model = gate_factory_config(cfg)
    x = torch.randn(12, 6)
    probs, mask, aux = model(x)
    assert mask.shape == probs.shape == (12, 8)
    assert (mask.sum(0) <= aux["capacity"]).all()
    (probs.square().mean() + aux["l_aux"]).backward()
    assert model.router.raw_gain.grad is not None
    restore = gate_factory_config(cfg)
    restore.load_state_dict(model.state_dict())
    torch.testing.assert_close(model.router(x), restore.router(x))


@pytest.mark.parametrize("arch", ["compact_born", "mps"])
def test_structured_fp32_island_under_cast_and_autocast(arch):
    model = RouterLogits(9, 8, gate_arch=arch, gate_norm="layernorm").to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = model(torch.randn(3, 9))
    assert y.dtype == torch.float32 and torch.isfinite(y).all()
    y.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


@pytest.mark.parametrize("e", [1, 3, 63, 128])
def test_born_rejects_invalid_expert_counts(e):
    with pytest.raises(ValueError):
        CompactBornGate(8, e)


@pytest.mark.parametrize("arch", ["compact_born", "mps"])
def test_structured_hf_save_reload(tmp_path, arch):
    from transformers import Qwen2Config
    from fitmotn.export.remote_code.configuration_fitmotn import FitMoTNConfig
    from fitmotn.export.remote_code.modeling_fitmotn import FitMoTNForCausalLM

    base = Qwen2Config(
        vocab_size=24,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    patch = dict(
        patch_backend="sparse_mixt",
        gate_type="topk",
        gate_arch=arch,
        gate_learnable_gain=True,
        gate_quantum=dict(rank=3, depth=2),
        gate_mps=dict(sites=3, bond_dim=4),
        E=2,
        topk=1,
        d=2,
        k_in=2,
        drop_tokens=False,
        pos_strategy="sliding",
        dtype="float32",
        sparse_mixt=dict(
            hidden_main=8, intermediate_main=16, router_input_policy="main"
        ),
    )
    config = FitMoTNConfig(
        base_model_name_or_path="tiny-local",
        base_model_config=base.to_dict(),
        fitmotn_patch_config=patch,
        layers_to_patch=[0],
        patch_backend="sparse_mixt",
    )
    model = FitMoTNForCausalLM(config).eval()
    ids = torch.tensor([[1, 3, 7, 5]])
    before = model(input_ids=ids, use_cache=False).logits
    before.square().mean().backward()
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = FitMoTNForCausalLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(
        before.detach(), restored(input_ids=ids, use_cache=False).logits
    )


def test_no_entangler_joint_distribution_factorizes():
    model = CompactBornGate(5, 8, entangle=False).double()
    a = torch.randn(2, model.depth, model.qubits, 2, dtype=torch.float64)
    p = model.probabilities_from_angles(a).reshape(2, 2, 2, 2)
    p0 = p.sum((2, 3))
    p1 = p.sum((1, 3))
    p2 = p.sum((1, 2))
    product = p0[:, :, None, None] * p1[:, None, :, None] * p2[:, None, None, :]
    torch.testing.assert_close(p, product, atol=1e-12, rtol=1e-12)
