from itertools import product
import numpy as np
import pytest
import torch
from fitmotn.advanced_routers import MPOOperator, ResidualMPOGate, AmplitudeReuploadBornGate
from fitmotn.gate import RouterLogits, gate_config_from_mapping, gate_factory_config
from fitmotn.experiments.gate_benchmark.config import load_config, validate
from fitmotn.experiments.gate_benchmark.models import router_config


def test_mpo_matches_independent_index_contraction_and_parameter_derivative():
    torch.manual_seed(6)
    mpo = MPOOperator(3, 2).double()
    cores = [p.detach().numpy() for p in mpo.cores]
    expected = np.zeros((8, 8))
    for out, inp in product(range(8), repeat=2):
        ob = [(out >> j) & 1 for j in (2, 1, 0)]
        ib = [(inp >> j) & 1 for j in (2, 1, 0)]
        for a, b in product(range(2), repeat=2):
            expected[out, inp] += (cores[0][0, ob[0], ib[0], a] *
                                  cores[1][a, ob[1], ib[1], b] *
                                  cores[2][b, ob[2], ib[2], 0])
    np.testing.assert_allclose(mpo.matrix().detach().numpy(), expected, atol=1e-12)
    x = torch.randn(2, 8, dtype=torch.double, requires_grad=True)
    torch.testing.assert_close(mpo(x), x @ torch.from_numpy(expected).T)
    assert torch.autograd.gradcheck(mpo, (x,))
    weight = torch.randn(8, 8, dtype=torch.double)
    (mpo.matrix()*weight).sum().backward()
    analytic = mpo.cores[1].grad[0, 1, 0, 1].item()
    with torch.no_grad():
        original = mpo.cores[1][0, 1, 0, 1].item()
        values = []
        for delta in (1e-6, -1e-6):
            mpo.cores[1][0, 1, 0, 1] = original+delta
            values.append((mpo.matrix()*weight).sum().item())
        mpo.cores[1][0, 1, 0, 1] = original
    assert analytic == pytest.approx((values[0]-values[1])/2e-6, rel=1e-7, abs=1e-8)


def test_amplitude_evolution_matches_independent_dense_numpy_and_preserves_norm():
    torch.manual_seed(31)
    gate = AmplitudeReuploadBornGate(5, 8, rank=3, depth=3).double()
    with torch.no_grad():
        gate.rotation.uniform_(-0.7, 0.7)
        gate.coupling.uniform_(-0.7, 0.7)
    state = torch.nn.functional.normalize(torch.randn(2, 8, dtype=torch.double), dim=-1)
    state.requires_grad_()
    angles = torch.randn(2, 3, 3, dtype=torch.double, requires_grad=True)
    actual = gate.evolve(state, angles)

    def kron(ops):
        result = np.ones((1, 1))
        for op in ops:
            result = np.kron(result, op)
        return result

    def ry(angle):
        c, s = np.cos(angle/2), np.sin(angle/2)
        return np.array([[c, -s], [s, c]])

    expected = []
    for initial, theta in zip(state.detach().numpy(), angles.detach().numpy()+gate.rotation.detach().numpy()):
        v = initial.copy()
        edge = 0
        for layer in range(3):
            for wire in range(3):
                v = kron([ry(theta[layer, wire]) if j == wire else np.eye(2) for j in range(3)]) @ v
            for wire in range(layer % 2, 2, 2):
                c, t = (wire, wire+1) if layer % 2 == 0 else (wire+1, wire)
                for control, target in ((c, t), (t, c)):
                    inactive = kron([np.diag([1, 0]) if j == control else np.eye(2) for j in range(3)])
                    active = kron([np.diag([0, 1]) if j == control else
                                   ry(gate.coupling[edge].item()) if j == target else np.eye(2)
                                   for j in range(3)])
                    v = (inactive+active) @ v
                    edge += 1
        expected.append(v)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-12)
    torch.testing.assert_close(actual.square().sum(-1), state.square().sum(-1), atol=2e-12, rtol=2e-12)
    assert torch.autograd.gradcheck(gate.evolve, (state, angles), atol=1e-5)


@pytest.mark.parametrize('cls', [ResidualMPOGate, AmplitudeReuploadBornGate])
@pytest.mark.parametrize('experts', [2, 16, 64])
def test_joint_shapes_all_parameter_gradients_and_updates(cls, experts):
    torch.manual_seed(27)
    model = cls(9, experts, rank=4, depth=2, norm='layernorm').double()
    x = torch.randn(2, 3, 9, dtype=torch.double, requires_grad=True)
    y = model(x)
    assert y.shape == (2, 3, experts) and torch.isfinite(y).all()
    if cls is AmplitudeReuploadBornGate:
        torch.testing.assert_close(y.exp().sum(-1), torch.ones(2, 3, dtype=torch.double))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    (y*torch.randn_like(y)).sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in model.parameters():
        if p.numel():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    optimizer.step()
    assert all(not torch.equal(p, before[n]) for n, p in model.named_parameters() if p.numel())


@pytest.mark.parametrize('arch', ['residual_mpo', 'amplitude_born'])
def test_factory_restore_fp32_island_and_input_gradient(arch):
    config = gate_config_from_mapping(dict(E=8, topk=2, gate_type='topk', gate_arch=arch,
        gate_learnable_gain=True, gate_norm='layernorm', drop_tokens=False,
        gate_mps=dict(rank=4, depth=2, bond_dim=2), gate_quantum=dict(rank=4, depth=2)), data_dim=9)
    gate = gate_factory_config(config)
    x = torch.randn(5, 9, requires_grad=True)
    probs, mask, aux = gate(x)
    assert (mask.sum(-1) == 2).all()
    (probs.square().sum()+aux['l_aux']).backward()
    assert torch.isfinite(x.grad).all()
    restored = gate_factory_config(config)
    restored.load_state_dict(gate.state_dict())
    torch.testing.assert_close(gate.router(x), restored.router(x))
    router = RouterLogits(9, 8, gate_arch=arch, gate_norm='layernorm').to(torch.bfloat16)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = router(torch.randn(3, 9))
    assert y.dtype == torch.float32 and torch.isfinite(y).all()


def test_variant_budget_matching_and_actual_half_budget():
    c = load_config()
    c['variants'] = {
        'mlp_half': {'arch': 'mlp', 'fraction': .5},
        'mpo_full': {'arch': 'residual_mpo', 'fraction': 1, 'bond_dim': 2},
        'mpo_half': {'arch': 'residual_mpo', 'fraction': .5, 'bond_dim': 2},
        'amp_full': {'arch': 'amplitude_born', 'fraction': 1},
        'amp_half': {'arch': 'amplitude_born', 'fraction': .5},
    }
    c['gates'] = ['mlp'] + list(c['variants'])
    validate(c)
    for d in [256, 1024, 2048]:
        counts = {}
        for variant in c['gates']:
            g = gate_factory_config(gate_config_from_mapping(router_config(c, variant, d, 0), data_dim=d))
            counts[variant] = sum(p.numel() for p in g.router.parameters())
        for variant, number in counts.items():
            fraction = .5 if 'half' in variant else 1
            assert abs(number/counts['mlp']-fraction) < .065


def test_mpo_can_represent_identity_and_bond_one_product():
    mpo = MPOOperator(4, 1).double()
    with torch.no_grad():
        for core in mpo.cores:
            core.copy_(torch.eye(2, dtype=torch.double).reshape(1, 2, 2, 1))
    torch.testing.assert_close(mpo.matrix(), torch.eye(16, dtype=torch.double))


@pytest.mark.parametrize('cls', [ResidualMPOGate, AmplitudeReuploadBornGate])
@pytest.mark.parametrize('experts', [1, 3, 128])
def test_invalid_joint_state_size_rejected(cls, experts):
    with pytest.raises(ValueError):
        cls(8, experts)
