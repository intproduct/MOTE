from __future__ import annotations

import torch

from fitmotn.train.trainer import (
    build_teacher_topk_targets,
    compute_topk_distillation_loss,
    compute_exact_distillation_loss,
)


def _labels() -> torch.Tensor:
    return torch.tensor([[-100, -100, 3, 4, 5]], dtype=torch.long)


def test_topk_kd_is_zero_for_identical_distributions():
    torch.manual_seed(7)
    teacher_logits = torch.randn(1, 5, 11)
    labels = _labels()
    targets = build_teacher_topk_targets(
        teacher_logits,
        labels,
        temperature=2.0,
        topk=4,
        token_chunk_size=2,
    )
    student_logits = teacher_logits.clone().requires_grad_(True)
    loss, metrics = compute_topk_distillation_loss(
        student_logits,
        labels,
        targets,
        temperature=2.0,
        token_chunk_size=2,
    )
    assert abs(float(loss.detach())) < 1e-5
    assert metrics["kd_valid_tokens"] == 3
    assert metrics["kd_topk"] == 4
    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_topk_kd_has_finite_nonzero_gradient_for_different_student():
    torch.manual_seed(11)
    teacher_logits = torch.randn(1, 5, 13)
    student_logits = torch.randn(1, 5, 13, requires_grad=True)
    labels = _labels()
    targets = build_teacher_topk_targets(
        teacher_logits,
        labels,
        temperature=1.5,
        topk=5,
        token_chunk_size=2,
    )
    loss, _ = compute_topk_distillation_loss(
        student_logits,
        labels,
        targets,
        temperature=1.5,
        token_chunk_size=2,
    )
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert torch.isfinite(student_logits.grad).all()
    assert float(student_logits.grad.abs().sum()) > 0.0


def test_topk_kd_ignores_prompt_only_positions():
    torch.manual_seed(23)
    teacher_logits = torch.randn(1, 5, 9)
    labels = _labels()
    targets = build_teacher_topk_targets(
        teacher_logits,
        labels,
        temperature=2.0,
        topk=3,
        token_chunk_size=4,
    )
    student_a = torch.randn(1, 5, 9)
    student_b = student_a.clone()
    student_b[:, 0, :] += 1000.0
    loss_a, _ = compute_topk_distillation_loss(
        student_a,
        labels,
        targets,
        temperature=2.0,
        token_chunk_size=4,
    )
    loss_b, _ = compute_topk_distillation_loss(
        student_b,
        labels,
        targets,
        temperature=2.0,
        token_chunk_size=4,
    )
    assert torch.allclose(loss_a, loss_b, atol=0.0, rtol=0.0)


def test_exact_kd_is_zero_for_identical_distributions():
    torch.manual_seed(31)
    teacher_logits = torch.randn(1, 5, 17)
    student_logits = teacher_logits.clone().requires_grad_(True)
    loss, metrics = compute_exact_distillation_loss(
        student_logits, teacher_logits, _labels(), temperature=2.0, token_chunk_size=2
    )
    assert abs(float(loss.detach())) < 1e-5
    assert metrics == {
        "kd_valid_tokens": 3,
        "kd_distribution": "exact",
        "kd_divergence": "forward_kl",
        "kd_jsd_beta": None,
    }
    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_exact_kd_has_gradient_and_ignores_prompt_only_positions():
    torch.manual_seed(37)
    teacher_logits = torch.randn(1, 5, 19)
    student_a = torch.randn(1, 5, 19, requires_grad=True)
    student_b = student_a.detach().clone()
    student_b[:, 0, :] += 1000.0
    loss_a, _ = compute_exact_distillation_loss(
        student_a, teacher_logits, _labels(), temperature=1.5, token_chunk_size=2
    )
    loss_b, _ = compute_exact_distillation_loss(
        student_b, teacher_logits, _labels(), temperature=1.5, token_chunk_size=2
    )
    assert float(loss_a.detach()) > 0.0
    assert torch.allclose(loss_a.detach(), loss_b, atol=1e-6, rtol=1e-6)
    loss_a.backward()
    assert torch.isfinite(student_a.grad).all()
    assert float(student_a.grad.abs().sum()) > 0.0

def test_exact_reverse_kl_and_jsd_are_zero_when_identical_and_have_gradients():
    torch.manual_seed(41)
    teacher_logits = torch.randn(1, 5, 23)
    labels = _labels()
    for divergence in ("reverse_kl", "jsd"):
        identical = teacher_logits.clone().requires_grad_(True)
        zero_loss, metrics = compute_exact_distillation_loss(
            identical,
            teacher_logits,
            labels,
            temperature=1.0,
            token_chunk_size=2,
            divergence=divergence,
            jsd_beta=0.5,
        )
        assert abs(float(zero_loss.detach())) < 1e-6
        assert metrics["kd_divergence"] == divergence

        student_logits = torch.randn(1, 5, 23, requires_grad=True)
        loss, _ = compute_exact_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            temperature=1.0,
            token_chunk_size=2,
            divergence=divergence,
            jsd_beta=0.5,
        )
        assert float(loss.detach()) > 0.0
        loss.backward()
        assert torch.isfinite(student_logits.grad).all()
        assert float(student_logits.grad.abs().sum()) > 0.0
