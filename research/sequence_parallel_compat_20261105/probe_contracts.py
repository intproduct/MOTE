"""Executable counterexamples for adapting the current Trainer and MoTE to SP.

All tensors are tiny CPU tensors; rank metadata checks are explicitly simulated.
These diagnostics preserve current code and expose integration work to be done.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from fitmotn.gate import TopKGate
from fitmotn.train.objectives import compute_loss_observability
from fitmotn.train.trainer import FitMoTNTrainer
from fitmotn.data.sampling import resolve_shard_context


def probe():
    torch.set_num_threads(1)
    results = {}
    logits = torch.tensor([[9., 0.], [8., 0.], [7., 0.], [6., 0.],
                           [0., 5.], [0., 4.], [0., 3.], [0., 2.]])
    gate = TopKGate(2, 2, k=1, min_capacity=1, capacity_factor=1., aux_coeff=1.)
    gate.router = torch.nn.Identity()
    _, full_mask, _ = gate(logits)
    parts = [gate(x) for x in logits.chunk(2)]
    local_mask = torch.cat([x[1] for x in parts])
    changed = (full_mask != local_mask).any(-1).sum().item()
    assert changed == 4
    results["capacity"] = dict(full_kept=full_mask.sum().item(),
                               shard_local_kept=local_mask.sum().item(), changed_tokens=changed)
    gate.drop_tokens = False
    _, _, full_aux = gate(logits)
    aux_mean = torch.stack([gate(x)[2]["l_aux"] for x in logits.chunk(2)]).mean()
    assert not torch.isclose(full_aux["l_aux"], aux_mean)
    results["aux_loss"] = dict(full=full_aux["l_aux"].item(), local_mean=aux_mean.item())

    # The repository objective shifts inside the local shard, losing boundaries.
    torch.manual_seed(17)
    token_logits = torch.randn(1, 8, 16)
    labels = torch.arange(1, 9).unsqueeze(0)
    full, _ = compute_loss_observability(token_logits, labels, collect_metrics=False)
    chunks = [compute_loss_observability(a, b, collect_metrics=False)[0]
              for a, b in zip(token_logits.chunk(2, 1), labels.chunk(2, 1))]
    local_mean = torch.stack(chunks).mean()
    shifted = F.pad(labels, (0, 1), value=-100)[:, 1:]
    corrected = sum(F.cross_entropy(a.reshape(-1, 16), b.reshape(-1), reduction="sum")
                    for a, b in zip(token_logits.chunk(2, 1), shifted.chunk(2, 1))) / 7
    torch.testing.assert_close(corrected, full)
    assert not torch.isclose(local_mean, full)
    results["label_boundary"] = dict(full_targets=7, locally_shifted_targets=6,
                                    full_loss=full.item(), local_shift_loss=local_mean.item(),
                                    pre_shift_loss=corrected.item())

    recorded = {}
    class Recorder(torch.nn.Module):
        def forward(self, **kwargs):
            recorded.update(kwargs)
            return SimpleNamespace(logits=token_logits, loss=None)
    actual = FitMoTNTrainer.compute_loss(
        SimpleNamespace(observability_state=None, kd_enabled=False), Recorder(),
        dict(input_ids=labels, position_ids=torch.arange(8).unsqueeze(0), shift_labels=shifted))
    assert "position_ids" not in recorded and actual is None
    results["official_batch_trainer_contract"] = dict(
        position_ids_forwarded=False, pre_shifted_label_loss_returned=False)

    shards = []
    for rank in (0, 1):
        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_rank", return_value=rank), \
             patch("torch.distributed.get_world_size", return_value=2):
            context = resolve_shard_context()
            shards.append([context.rank, context.world_size])
    results["sampler_simulated_rank_metadata"] = shards
    try:
        FitMoTNTrainer._save_checkpoint(
            SimpleNamespace(checkpoint_policy=object(), args=SimpleNamespace(world_size=2)), None, None)
    except RuntimeError as exc:
        results["checkpoint_guard"] = str(exc)
    else:
        raise AssertionError("Expected the current multi-process checkpoint guard")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))
