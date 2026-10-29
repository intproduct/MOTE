"""Numerical checks of mathematical claims, NOT model-quality benchmarks.

Run with Python + NumPy. Results are written next to this script.
"""

import json
from pathlib import Path

import numpy as np


def softmax(x):
    y = np.exp(x - np.max(x))
    return y / y.sum()


def main():
    rng = np.random.default_rng(20261029)
    p = rng.uniform(0.01, 1, size=16)
    p /= p.sum()
    a = rng.normal(size=16)
    born = a**2 / np.dot(a, a)
    eps, gain, temperature = 1e-5, 1.7, 0.8
    smoothed = (born + eps) / (1 + len(a) * eps)
    power = smoothed ** (gain / temperature)
    power /= power.sum()

    # Tangent vectors to a normalized real state, with nonzero amplitudes.
    psi = a / np.linalg.norm(a)
    tangent = rng.normal(size=(16, 4))
    tangent -= psi[:, None] * (psi @ tangent)[None, :]
    dp = 2 * psi[:, None] * tangent
    cfi = dp.T @ (dp / (psi**2)[:, None])
    qfi = 4 * (tangent.T @ tangent)

    # A phase-only complex state has nonzero QFI but constant Born output.
    theta = 0.37
    phase_state = np.array([1, np.exp(1j * theta)]) / np.sqrt(2)
    phase_deriv = np.array([0, 1j * np.exp(1j * theta)]) / np.sqrt(2)
    phase_dp = 2 * np.real(np.conj(phase_state) * phase_deriv)
    phase_cfi = np.sum(phase_dp**2 / np.abs(phase_state) ** 2)
    phase_qfi = 4 * (
        np.vdot(phase_deriv, phase_deriv).real
        - abs(np.vdot(phase_state, phase_deriv)) ** 2
    )

    ortho, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    b = rng.normal(size=16)
    distance_error = abs(np.linalg.norm(ortho @ a - ortho @ b) - np.linalg.norm(a - b))

    # Tiny probability change can still swap hard top-1 near a tie.
    delta = 1e-8
    p_left = np.array([0.5 + delta, 0.5 - delta])
    p_right = p_left[::-1]

    # Classical cosine-logit CE has the same nonlinear curvature term.
    x, label = 0.21, 1.0
    angle = np.pi * x + theta
    logit = np.cos(angle)
    pred = 1 / (1 + np.exp(-logit))
    analytic_hessian = pred * (1 - pred) * np.sin(angle) ** 2 - (pred - label) * np.cos(angle)

    def loss(t):
        z = np.cos(np.pi * x + t)
        return np.logaddexp(0, z) - label * z

    h = 1e-4
    numeric_hessian = (loss(theta + h) - 2 * loss(theta) + loss(theta - h)) / h**2

    results = {
        "scope": "Mathematical sanity checks only; no model training or performance evidence.",
        "seed": 20261029,
        "numpy_version": np.__version__,
        "simplex_sqrt_born_max_error": float(np.max(np.abs(np.sqrt(p) ** 2 - p))),
        "simplex_log_softmax_max_error": float(np.max(np.abs(softmax(np.log(p)) - p))),
        "squared_amplitude_softmax_max_error": float(np.max(np.abs(softmax(2 * np.log(np.abs(a))) - born))),
        "gain_temperature_power_max_error": float(np.max(np.abs(softmax(gain / temperature * np.log(smoothed)) - power))),
        "real_state_cfi_qfi_max_error": float(np.max(np.abs(cfi - qfi))),
        "phase_only_classical_fisher": float(phase_cfi),
        "phase_only_quantum_fisher": float(phase_qfi),
        "fixed_orthogonal_distance_error": float(distance_error),
        "near_tie_total_variation": float(0.5 * np.abs(p_left - p_right).sum()),
        "near_tie_top1_changed": bool(np.argmax(p_left) != np.argmax(p_right)),
        "classical_cosine_hessian_absolute_error": float(abs(analytic_hessian - numeric_hessian)),
    }
    assert max(results[k] for k in (
        "simplex_sqrt_born_max_error", "simplex_log_softmax_max_error",
        "squared_amplitude_softmax_max_error", "gain_temperature_power_max_error",
        "real_state_cfi_qfi_max_error", "fixed_orthogonal_distance_error",
    )) < 1e-10
    assert abs(phase_qfi - 1) < 1e-12 and phase_cfi < 1e-12
    assert results["near_tie_top1_changed"]
    assert results["classical_cosine_hessian_absolute_error"] < 1e-6
    results["all_checks_passed"] = True
    output = Path(__file__).with_name("geometry_checks.json")
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
