from __future__ import annotations
import json
import numpy as np
import pytest
import torch
from fitmotn.experiments.gate_benchmark.config import load_config
from fitmotn.experiments.gate_benchmark.__main__ import tiny_config, smoke
from fitmotn.experiments.gate_benchmark.data import split_train, stratified_indices
from fitmotn.experiments.gate_benchmark.models import (
    ImageClassifier,
    BertClassifier,
    shared_state,
    load_shared,
    state_hash,
    auxiliary_loss,
    routed_modules,
    warm_initialize,
    router_config,
)
from fitmotn.experiments.gate_benchmark.runner import metrics, seed_all
from fitmotn.experiments.gate_benchmark.report import summarize


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_split_fixed_disjoint_stratified():
    cfg = load_config()
    labels = np.array([0] * 100 + [1] * 30)
    train, dev = split_train(labels, cfg)
    assert not set(train) & set(dev) and len(train) + len(dev) == 130
    assert (train, dev) == split_train(labels, cfg)
    assert len(stratified_indices(labels, 37, 5)) == 37
    assert len(set(train)) == len(train)


@pytest.mark.parametrize("layers", [2, 3])
@pytest.mark.parametrize("arch", ["mlp", "compact_born", "mps"])
def test_bert_padding_aux_backward(layers, arch):
    cfg, bert = tiny_config(load_config(), "bert")
    cfg["bert_layers"] = layers
    bert.num_hidden_layers = layers
    model = BertClassifier(cfg, arch, 11, 2, tiny_config=bert).eval()
    ids = torch.randint(1, 40, (2, 8))
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    y = model(input_ids=ids, attention_mask=mask)
    assert len(routed_modules(model)) == 2 * layers
    assert all(m.last_tokens == 10 for _, m in routed_modules(model))
    changed = ids.clone()
    changed[:, -3:] = torch.randint(1, 40, (2, 3))
    torch.testing.assert_close(y, model(input_ids=changed, attention_mask=mask))
    (y.square().mean() + auxiliary_loss(model)).backward()
    for _, m in routed_modules(model):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in m.mote.core.gate.router.parameters()
        )
        assert m.attention_mask is None


def test_shared_experts_identical_for_different_routers():
    cfg, _ = tiny_config(load_config(), "mlp")
    reference = ImageClassifier(cfg, "mlp", 17, 32, 2)
    shared = shared_state(reference)
    expected = state_hash(shared)
    for arch in ("compact_born", "mps"):
        model = ImageClassifier(cfg, arch, 17, 32, 2)
        load_shared(model, shared)
        assert state_hash(shared_state(model)) == expected
        for (_, a), (_, b) in zip(routed_modules(reference), routed_modules(model)):
            assert a.mote.core.positions == b.mote.core.positions


def test_common_warm_initialization_reduces_probe_error():
    cfg, bert = tiny_config(load_config(), "bert")
    seed_all(3)
    model = BertClassifier(cfg, "mlp", 3, 2, tiny_config=bert)
    report = warm_initialize(model, 30, 4)
    assert report and all(
        v["relative_mse_after"] < v["relative_mse_before"] for v in report.values()
    )
    assert all(m.source_weight is None for _, m in routed_modules(model))


def test_router_parameter_budget_matching():
    from fitmotn.gate import gate_config_from_mapping, gate_factory_config

    c = load_config()
    for d in (256, 1024, 2048):
        counts = []
        for arch in c["gates"]:
            gate = gate_factory_config(
                gate_config_from_mapping(router_config(c, arch, d, 0), data_dim=d)
            )
            counts.append(sum(p.numel() for p in gate.router.parameters()))
        assert (max(counts) - min(counts)) / min(counts) < 0.06


def test_mrpc_metric_known_values():
    r = metrics(
        torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([1, 0, 1, 0]),
        "mrpc",
    )
    assert r["accuracy"] == r["f1"] == r["score"] == 0.5


def test_offline_six_run_smoke_and_summary(tmp_path):
    cfg = load_config()
    cfg["device"] = "cpu"
    cfg["seeds"] = [7]
    path = smoke(cfg, str(tmp_path / "runs"))
    assert len(list(path.rglob("result.json"))) == 6
    groups = summarize(path)
    assert len(groups) == 6
    assert all(g["paired_n"] == 1 for g in groups)
    assert (path / "runs.csv").exists()
    for p in path.rglob("result.json"):
        data = json.loads(p.read_text())
        assert data["best_step"] in (1, 2)


def test_prepare_cache_and_run_subset_reproducible(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fitmotn.experiments.gate_benchmark.data import prepare, load_data
    from fitmotn.experiments.gate_benchmark.runner import run
    import datasets
    import huggingface_hub

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows
            self.column_names = ["image", "label"]
            self._fingerprint = str(len(rows))

        def __getitem__(self, key):
            return [r[key] for r in self.rows]

        def select(self, idx):
            return FakeDataset([self.rows[i] for i in idx])

    rows = [
        {"image": np.full((4, 8), i % 10, dtype=np.uint8), "label": i % 2}
        for i in range(30)
    ]
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, **kw: {"train": FakeDataset(rows), "test": FakeDataset(rows[:12])},
    )
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(
            dataset_info=lambda repo: SimpleNamespace(sha="fake-revision")
        ),
    )
    c, _ = tiny_config(load_config(), "mlp")
    c.update(
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "out"),
        device="cpu",
        seeds=[17],
    )
    c["mlp"]["tasks"] = ["mnist"]
    c["bert"]["tasks"] = []
    prepare(c, "mlp")
    data, meta = load_data(c, "mnist")
    assert meta["revision"] == "fake-revision"
    assert not set(meta["indices"]["train"]) & set(meta["indices"]["dev"])
    assert data["train"]["features"].shape[1] == 32
    path = run(c, "mlp", seeds=[17], gates=["mlp"])
    run(c, "mlp", seeds=[17], gates=["compact_born", "mps"])
    run(c, "mlp")  # completed runs are skipped
    groups = summarize(path)
    assert len(groups) == 3 and all(g["paired_n"] == 1 for g in groups)
    summary = json.loads((path / "summary.json").read_text())
    assert not summary["missing_runs"] and not summary["incomplete_runs"]


def test_64_expert_real_width_mote_backwards():
    from fitmotn.experiments.gate_benchmark.models import MoTELinear

    c = load_config()
    c.update(experts=64, topk=8)
    for arch in c["gates"]:
        model = MoTELinear(256, 1024, c, arch, 11)
        x = torch.randn(2, 256, requires_grad=True)
        y = model(x)
        (y.square().mean() + model.last_aux["l_aux"]).backward()
        assert y.shape == (2, 1024) and torch.isfinite(x.grad).all()
        assert model.mote.core.num_blocks == 64
