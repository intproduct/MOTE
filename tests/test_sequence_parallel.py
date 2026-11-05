from types import SimpleNamespace

import pytest
import torch

from fitmotn.train.sequence_parallel import UlyssesConfig, prepare_batch, validate_model


def test_global_shift_weights_and_positions():
    batch = dict(input_ids=torch.arange(8).reshape(1, 8), labels=torch.arange(8).reshape(1, 8),
                 loss_weights=torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.]]))
    result = prepare_batch(batch, 8)
    assert result["shift_loss_weights"].tolist() == [[2., 3., 4., 5., 6., 7., 8., 0.]]
    assert result["position_ids"].tolist() == [list(range(8))]
    assert "loss_weights" in batch  # no caller mutation


@pytest.mark.parametrize("field,value", [
    ("attention_mask", torch.tensor([[1, 1, 1, 0]])),
    ("position_ids", torch.tensor([[0, 1, 0, 1]])),
    ("loss_weights", torch.tensor([[1., float("nan"), 1., 1.]])),
    ("loss_weights", torch.tensor([[1., -1., 1., 1.]])),
])
def test_reject_semantic_changes(field, value):
    batch = dict(input_ids=torch.ones(1, 4, dtype=torch.long), labels=torch.ones(1, 4, dtype=torch.long))
    batch[field] = value
    with pytest.raises(ValueError):
        prepare_batch(batch, 4)


@pytest.mark.parametrize("family", ["gemma3", "qwen3_5", "unknown"])
def test_reject_unadapted_architectures(family):
    with pytest.raises(ValueError, match="separate adapters"):
        validate_model(SimpleNamespace(config=SimpleNamespace(model_type=family)), UlyssesConfig(2, 16))


@pytest.mark.parametrize("all_ignored", [False, True])
def test_zero_weights_fallback_and_empty_supervision(monkeypatch, all_ignored):
    from fitmotn.train.sequence_parallel import UlyssesRuntime
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group: tensor.mul_(2))
    logits = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
    labels = torch.tensor([[-100, -100] if all_ignored else [0, 1]])
    runtime = SimpleNamespace(auxiliary=[], config=UlyssesConfig(2, 4), group=None)
    inputs = dict(input_ids=torch.ones(1, 2, dtype=torch.long), position_ids=torch.arange(2)[None],
                  shift_labels=labels, shift_loss_weights=torch.zeros(1, 2))
    loss = UlyssesRuntime.compute_loss(runtime, lambda **kw: SimpleNamespace(logits=logits), inputs)
    expected = 0. if all_ignored else torch.nn.functional.cross_entropy(logits.reshape(-1, 2), labels.flatten()).item()
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_tokenized_dataset_fingerprint_and_mutation(tmp_path):
    from fitmotn.data.tokenized_sequences import TokenizedSequenceDataset
    path = tmp_path / "tokens.jsonl"
    path.write_text('{"input_ids":[1,2],"labels":[-100,2]}\n', encoding="utf-8")
    dataset = TokenizedSequenceDataset(path)
    assert len(dataset) == 1 and len(dataset.fingerprint) == 64
    assert dataset[0]["labels"].tolist() == [-100, 2]
    path.write_text('{"input_ids":[1,2,3],"labels":[-100,2,3]}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed during training"):
        dataset[0]
