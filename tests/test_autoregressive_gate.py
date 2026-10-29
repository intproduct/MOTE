import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fitmotn.experiments.autoregressive_gate.data import (
    training_batch, evaluation_batches, articles,
)
from fitmotn.experiments.autoregressive_gate.model import initialized_model
from fitmotn.experiments.autoregressive_gate.runner import evaluate
from fitmotn.experiments.gate_benchmark.models import (
    state_hash, shared_state, parameter_report, auxiliary_loss, is_router_key,
)


def configuration():
    path = Path(__file__).parents[1]/'experiments/autoregressive_gate/configs/pilot.json'
    cfg = json.loads(path.read_text())
    cfg['device'] = 'cpu'
    cfg['model'].update(dim=128, ff_dim=256, layers=1, context=12, vocab_size=37)
    cfg['eval'].update(stride=5, batch_size=3)
    return cfg


@pytest.mark.parametrize('variant', configuration()['gates'])
def test_causality_batch_independence_and_padding(variant):
    torch.set_num_threads(1)
    cfg = configuration()
    model, _ = initialized_model(cfg, variant, 17)
    model.eval()
    x = torch.randint(2, 37, (2, 12))
    with torch.no_grad():
        before = model(x)
        modified = x.clone()
        modified[:, 6:] = (modified[:, 6:]+1)%37
        after = model(modified)
        torch.testing.assert_close(before[:, :6], after[:, :6], rtol=0, atol=1e-6)
        solo = model(x[:1])
        torch.testing.assert_close(before[:1], solo, rtol=1e-5, atol=1e-6)
        valid = torch.arange(12)[None, :].expand(2, -1) < 6
        padded = model(modified, valid)
        torch.testing.assert_close(before[:, :6], padded[:, :6], rtol=1e-5, atol=1e-6)


def test_shared_initialization_budgets_and_real_router_updates(tmp_path):
    torch.set_num_threads(1)
    cfg = configuration()
    hashes, counts = set(), {}
    for variant in cfg['gates']:
        model, initial = initialized_model(cfg, variant, 17)
        hashes.add(state_hash(initial))
        assert state_hash(shared_state(model)) == state_hash(initial)
        counts[variant] = parameter_report(model)['router_total']
        before = {n: p.detach().clone() for n,p in model.named_parameters() if is_router_key(n)}
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        x = torch.randint(2, 37, (2, 12))
        y = torch.randint(2, 37, (2, 12))
        for _ in range(3):
            opt.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), y.flatten()) + auxiliary_loss(model)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
        changes = {n: (p.detach()-before[n]).abs().max().item() for n,p in model.named_parameters() if n in before}
        assert all(v > 0 for v in changes.values()), changes
        path = tmp_path/f'{variant}.pt'
        torch.save(model.state_dict(), path)
        restored, _ = initialized_model(cfg, variant, 17)
        restored.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
    assert len(hashes) == 1
    for v in ['mpo_c2_full','amp_d1_full']:
        assert abs(counts[v]/counts['mlp']-1) < .05
    for v in ['mlp_half','mpo_c1_half','amp_d1_half']:
        assert abs(counts[v]/counts['mlp']-.5) < .05


@pytest.mark.parametrize('length', [2, 6, 13, 14, 19, 43, 100])
@pytest.mark.parametrize('stride', [1, 5, 12])
def test_strided_evaluation_scores_every_target_once(length, stride):
    tokens = torch.arange(length)
    seen = []
    for x, y, valid in evaluation_batches(tokens, 12, stride, 3, 'cpu'):
        assert torch.all(y[~valid] == -100)
        active = y != -100
        torch.testing.assert_close(y[active], x[active]+1)
        seen.extend(y[active].tolist())
    assert seen == list(range(1, length))


def test_training_shift_and_generator_independence():
    data = torch.arange(100)
    a = torch.Generator().manual_seed(42)
    b = torch.Generator().manual_seed(42)
    x, y, start = training_batch(data, 4, 12, a, 'cpu')
    torch.randn(30)  # unrelated model RNG must not change minibatch selection
    x2, y2, start2 = training_batch(data, 4, 12, b, 'cpu')
    torch.testing.assert_close(y, x+1)
    torch.testing.assert_close(x, x2)
    torch.testing.assert_close(start, start2)


def test_token_weighted_cross_entropy_is_batch_invariant():
    class PositionFreeModel(torch.nn.Module):
        def forward(self, x, valid=None):
            p = torch.arange(1, 38, dtype=torch.float64).log()
            return p.expand(*x.shape, 37)
    cfg = configuration()
    tokens = torch.arange(25)
    result = evaluate(PositionFreeModel(), tokens, cfg, 'cpu')
    expected = -(torch.arange(2, 26, dtype=torch.float64)/sum(range(1,38))).log().mean().item()
    assert result['tokens'] == 24
    assert result['ce'] == pytest.approx(expected, abs=1e-12)
    assert result['ppl'] == pytest.approx(np.exp(expected), abs=1e-12)
    cfg['eval']['batch_size'] = 1
    assert evaluate(PositionFreeModel(), tokens, cfg, 'cpu')['ce'] == pytest.approx(expected, abs=1e-12)


def test_article_split_keeps_subheadings_and_original_text():
    lines = ['\n', ' = First article = \n', 'Text.\n', ' = = Section = = \n', 'More.\n',
             '\n', ' = Second article = \n', 'Other.\n']
    found = articles(lines)
    assert len(found) == 2
    assert 'Section' in found[0]
    assert ''.join(found) == ''.join(lines[1:])
