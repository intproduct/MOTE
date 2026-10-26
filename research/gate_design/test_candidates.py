import _bootstrap  # noqa: F401
import math

import numpy as np
import pytest
import torch

from candidates import BrickwallConeGate, SweepDensityGate, ResidualMPSGate, CompactBornGate
from fitmotn.gate import TopKGate


def dense_reference(model, angles):
    """Independent dense NumPy simulation; never used by research candidates."""
    n = model.num_experts
    eye, p0, p1 = np.eye(2), np.diag([1, 0]), np.diag([0, 1])
    def ry(a):
        c, s = np.cos(a / 2), np.sin(a / 2)
        return np.array([[c, -s], [s, c]])
    def product(ops):
        result = np.ones((1, 1))
        for op in ops:
            result = np.kron(result, op)
        return result
    def rotate(state, a, q):
        return product([ry(a) if i == q else eye for i in range(n)]) @ state
    def controlled(state, a, c, t):
        inactive = product([p0 if i == c else eye for i in range(n)])
        active = product([p1 if i == c else ry(a) if i == t else eye for i in range(n)])
        return (inactive + active) @ state
    values = []
    directions = 1 if isinstance(model, BrickwallConeGate) else model.directions
    for direction in range(directions):
        state = np.zeros(2**n)
        state[0] = 1
        if isinstance(model, BrickwallConeGate):
            edge = 0
            for layer in range(model.depth):
                for q in range(n):
                    theta = model.encoding_scale[layer, q].item() * angles[q] + model.rotation[layer, q].item()
                    state = rotate(state, theta, q)
                for q in range(layer % 2, n - 1, 2):
                    c, t = (q, q + 1) if layer % 2 == 0 else (q + 1, q)
                    for control, target in ((c, t), (t, c)):
                        state = controlled(state, model.coupling[edge].item(), control, target)
                        edge += 1
            axes = model.readout_rotation.detach().numpy()
        else:
            for q in range(n):
                state = rotate(state, angles[q] + model.rotation[direction, q].item(), q)
            order = list(range(n))[::1 if direction == 0 else -1]
            for q, nxt in zip(order[:-1], order[1:]):
                edge = min(q, nxt)
                state = controlled(state, model.coupling[direction, edge, 0].item(), q, nxt)
                state = controlled(state, model.coupling[direction, edge, 1].item(), nxt, q)
            axes = model.readout_rotation[direction].detach().numpy()
        for q in range(n):
            state = rotate(state, axes[q], q)
        np.testing.assert_allclose(state @ state, 1., atol=1e-12)
        values.append([state @ product([np.diag([1, -1]) if i == q else eye for i in range(n)]) @ state for q in range(n)])
    return np.mean(values, axis=0)


@pytest.mark.parametrize("n,depth", [(1, 1), (3, 2), (5, 2), (6, 3), (5, 4)])
def test_brickwall_exact_contraction_matches_dense(n, depth):
    torch.manual_seed(17)
    model = BrickwallConeGate(5, n, rank=3, depth=depth).double()
    with torch.no_grad():
        model.coupling.uniform_(-1.2, 1.2)
        model.rotation.uniform_(-1., 1.)
    angles = torch.randn(3, n, dtype=torch.float64)
    expected = np.stack([dense_reference(model, a.numpy()) for a in angles])
    np.testing.assert_allclose(model.expectations_from_angles(angles).detach().numpy(), expected, atol=3e-12)
    assert model.max_cone_qubits <= min(n, 2 * depth)


@pytest.mark.parametrize("n,bidirectional", [(1, True), (3, False), (4, True), (6, True)])
def test_frontier_density_contraction_matches_full_quantum_state(n, bidirectional):
    torch.manual_seed(19)
    model = SweepDensityGate(5, n, rank=3, bidirectional=bidirectional).double()
    with torch.no_grad():
        model.coupling.uniform_(-1.4, 1.4)
        model.readout_rotation.uniform_(-1., 1.)
    angles = torch.randn(3, n, dtype=torch.float64)
    expected = np.stack([dense_reference(model, a.numpy()) for a in angles])
    np.testing.assert_allclose(model.expectations_from_angles(angles).detach().numpy(), expected, atol=3e-12)


@pytest.mark.parametrize("factory", [BrickwallConeGate, SweepDensityGate])
def test_circuit_input_and_parameter_gradients(factory):
    torch.manual_seed(23)
    model = factory(4, 5, rank=3).double()
    angles = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(model.expectations_from_angles, (angles,), atol=1e-5)
    weights = torch.tensor([1., -0.7, 0.3, 1.2, -1.], dtype=torch.float64)
    objective = lambda: (model.expectations_from_angles(angles) * weights).sum()
    objective().backward()
    for parameter in (model.coupling, model.rotation, model.readout_rotation):
        index = parameter.numel() // 2
        old = parameter.detach().flatten()[index].item()
        grad = parameter.grad.flatten()[index].item()
        with torch.no_grad():
            parameter.flatten()[index] = old + 1e-6
            plus = objective().item()
            parameter.flatten()[index] = old - 1e-6
            minus = objective().item()
            parameter.flatten()[index] = old
        assert grad == pytest.approx((plus - minus) / 2e-6, abs=2e-6)


def test_mps_is_normalized_matrix_product_not_an_uncontrolled_recurrence():
    model = ResidualMPSGate(4, 5, sites=3, bond_dim=4).double()
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(model, (x,), atol=1e-5)
    with torch.no_grad():
        values = torch.tanh(model.input_proj(x))
        expected = []
        for row in values:
            state = model.boundary.clone()
            for i in range(model.sites):
                state = state @ (torch.eye(4) + row[i] * model.cores[i] / math.sqrt(model.sites))
            state = torch.nn.functional.normalize(state, dim=-1)
            expected.append(model.output_proj(state) * torch.nn.functional.softplus(model.raw_gain))
        torch.testing.assert_close(model(x), torch.stack(expected))


@pytest.mark.parametrize("factory", [BrickwallConeGate, SweepDensityGate, ResidualMPSGate, CompactBornGate])
def test_router_contract_and_optimizer_steps(factory):
    torch.manual_seed(29)
    model = factory(9, 5)
    gate = TopKGate(9, 5, k=2, drop_tokens=False, aux_coeff=0.03)
    gate.router = model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, targets = torch.randn(12, 9), torch.randn(12, 5)
    for _ in range(2):
        optimizer.zero_grad()
        probs, mask, aux = gate(x)
        torch.testing.assert_close(probs.sum(-1), torch.ones(12))
        assert (mask.sum(-1) == 2).all()
        loss = ((probs - targets)**2).mean() + aux["l_aux"]
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        optimizer.step()


@pytest.mark.parametrize("factory", [BrickwallConeGate, SweepDensityGate, ResidualMPSGate, CompactBornGate])
def test_cast_low_precision_and_state_dict_roundtrip(factory):
    model = factory(9, 5).to(torch.bfloat16)
    x = torch.randn(2, 3, 9, dtype=torch.bfloat16)
    y = model(x)
    assert y.shape == (2, 3, 5) and y.dtype == torch.float32
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    restored = factory(9, 5).to(torch.bfloat16)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), y)


def test_width64_does_not_allocate_a_64_qubit_state():
    for model in [BrickwallConeGate(8, 64, depth=2), SweepDensityGate(8, 64)]:
        angles = torch.randn(2, 64)
        z = model.expectations_from_angles(angles)
        assert z.shape == angles.shape and z.abs().max() <= 1.00001
    assert model.directions == 2
    assert BrickwallConeGate(8, 64, depth=2).max_cone_qubits == 4


def test_connected_brickwall_readout_depends_on_a_different_original_pair():
    model = BrickwallConeGate(5, 6, depth=2).double()
    with torch.no_grad():
        model.coupling.fill_(1.1)
    a = torch.tensor([0.4, -0.3, 0.5, -0.7, 0.2, 0.6], dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(model.expectations_from_angles, a)
    assert jacobian[2, 0].abs() > 1e-5
    assert jacobian[2, 5].abs() < 1e-12


@pytest.mark.parametrize("experts", [3, 16, 64])
def test_compact_joint_born_reference_and_gradient(experts):
    model = CompactBornGate(5, experts, rank=3).double()
    q = model.qubits
    angles = torch.randn(2, model.depth, q, 2, dtype=torch.float64, requires_grad=True)
    probs = model.probabilities_from_angles(angles)
    assert probs.shape == (2, 2**q) and 2**q < 2 * experts
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, dtype=torch.float64))
    if experts == 3:
        assert torch.autograd.gradcheck(model.probabilities_from_angles, (angles,), atol=1e-5)
    # Explicit NumPy statevector with independent dense complex operators.
    def kron(ops):
        out = np.ones((1, 1), dtype=np.complex128)
        for op in ops:
            out = np.kron(out, op)
        return out
    def ry(a):
        return np.array([[np.cos(a/2), -np.sin(a/2)], [np.sin(a/2), np.cos(a/2)]])
    eye = np.eye(2)
    expected = []
    for item in angles.detach().numpy():
        state = np.zeros(2**q, dtype=np.complex128)
        state[0] = 1
        theta = item + model.rotation.detach().numpy()
        edge = 0
        for layer in range(model.depth):
            for wire in range(q):
                state = kron([ry(theta[layer, wire, 0]) if j == wire else eye for j in range(q)]) @ state
                rz = np.diag(np.exp(np.array([-0.5j, 0.5j]) * theta[layer, wire, 1]))
                state = kron([rz if j == wire else eye for j in range(q)]) @ state
            for wire in range(layer % 2, q - 1, 2):
                c, t = (wire, wire + 1) if layer % 2 == 0 else (wire + 1, wire)
                for control, target in ((c, t), (t, c)):
                    inactive = kron([np.diag([1, 0]) if j == control else eye for j in range(q)])
                    active = kron([np.diag([0, 1]) if j == control else ry(model.coupling[edge].item()) if j == target else eye for j in range(q)])
                    state = (inactive + active) @ state
                    edge += 1
        for wire in range(q):
            state = kron([ry(model.readout_rotation[wire].item()) if j == wire else eye for j in range(q)]) @ state
        expected.append(np.abs(state)**2)
    np.testing.assert_allclose(probs.detach().numpy(), expected, atol=3e-12)
