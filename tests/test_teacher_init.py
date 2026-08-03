import torch
import torch.nn as nn

from fitmotn.init.teacher import evaluate_teacher_ffn, fit_teacher_ffn


def test_evaluate_teacher_ffn_reports_exact_match():
    module = nn.Linear(4, 3, bias=False)
    inputs = torch.randn(11, 4)
    targets = module(inputs).detach()

    metrics = evaluate_teacher_ffn(
        module,
        inputs,
        targets,
        device=torch.device("cpu"),
        batch_tokens=4,
    )

    assert metrics["mse"] == 0.0
    assert metrics["rel_l2"] == 0.0
    assert metrics["cosine"] > 0.999999
    assert metrics["tokens"] == 11.0


def test_fit_teacher_ffn_improves_validation_and_restores_best_state():
    torch.manual_seed(7)
    teacher = nn.Sequential(nn.Linear(5, 8), nn.SiLU(), nn.Linear(8, 3))
    student = nn.Sequential(nn.Linear(5, 8), nn.SiLU(), nn.Linear(8, 3))
    inputs = torch.randn(96, 5)
    targets = teacher(inputs).detach()
    records = []

    result, best_state = fit_teacher_ffn(
        student,
        inputs[:72],
        targets[:72],
        inputs[72:],
        targets[72:],
        device=torch.device("cpu"),
        steps=80,
        batch_tokens=24,
        eval_batch_tokens=8,
        eval_every=20,
        lr=1e-2,
        seed=11,
        progress=records.append,
    )

    restored = evaluate_teacher_ffn(
        student,
        inputs[72:],
        targets[72:],
        device=torch.device("cpu"),
        batch_tokens=8,
    )
    assert result["best"]["rel_l2"] < result["initial"]["rel_l2"]
    assert restored["rel_l2"] == result["best"]["rel_l2"]
    assert result["best_step"] in {20, 40, 60, 80}
    assert len(records) == 4
    assert set(best_state) == set(student.state_dict())
