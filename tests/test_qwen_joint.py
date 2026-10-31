"""Math, dispatch, conversion, actual Qwen gradients, and interruption replay."""
import copy
import random

import numpy as np
import pytest
import torch

from fitmotn.experiments.qwen_joint.routers import METHODS, JointRouter, hard_topk, reencoded_state
from fitmotn.experiments.qwen_joint.model import (AdapterConfig, JointProjection, auxiliary_loss,
    dense_from_tensor, patch_qwen, project_dense, tensor_hash)
from fitmotn.experiments.qwen_joint.state import MasterAdamW, TokenSchedule, load_checkpoint, save_checkpoint
from fitmotn.quantum_gate import apply_ry


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("entangle,rank", [(False, 4), (True, 8)])
def test_state_norm_rank_product_collapse_and_gradcheck(entangle, rank):
    g = torch.Generator().manual_seed(733)
    for _ in range(3):
        a = (torch.rand(8, generator=g, dtype=torch.float64) * 2 + 0.2).requires_grad_()
        state = reencoded_state(a[None], entangle)
        torch.testing.assert_close(state.square().sum(), torch.tensor(1., dtype=torch.float64), atol=1e-12, rtol=0)
        jac = torch.autograd.functional.jacobian(lambda v: reencoded_state(v[None], entangle)[0].square(), a)
        assert int(torch.linalg.matrix_rank(jac, atol=1e-8)) == rank
        if not entangle:
            collapsed = torch.zeros(1, 16, dtype=a.dtype)
            collapsed[:, 0] = 1
            for wire in range(4):
                collapsed = apply_ry(collapsed, (a[wire] + a[wire + 4])[None], wire, 4)
            torch.testing.assert_close(collapsed, state, atol=1e-12, rtol=0)
    assert torch.autograd.gradcheck(lambda v: reencoded_state(v, entangle), (a[None],))


def test_topk_sign_ties_normalization_and_unselected_gradient():
    logits = torch.tensor([[4., 4., 3., 2., 1.]], requires_grad=True)
    route = hard_topk(logits, 2)
    assert route.indices.tolist() == [[0, 1]]
    torch.testing.assert_close(route.weights, torch.tensor([[0.5, 0.5]]))
    (route.weights * torch.tensor([[1., 3.]])).sum().backward()
    assert logits.grad[0, :2].abs().min() > 0
    assert torch.equal(logits.grad[0, 2:], torch.zeros(3))


@pytest.mark.parametrize("method", METHODS)
def test_router_contract_precision_and_gradients(method):
    router = JointRouter(64, method, 17)
    x = torch.randn(5, 64, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        route = router(x)
    assert route.weights.dtype == route.probabilities.dtype == torch.float32
    torch.testing.assert_close(route.weights.sum(-1), torch.ones(5))
    torch.testing.assert_close(route.probabilities.sum(-1), torch.ones(5))
    loss = (route.weights * torch.arange(4)).sum() + route.load_balance * 0.01
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in router.parameters())
    assert router.input_proj.weight.grad.norm() > 0
    assert x.grad.norm() > 0


def test_common_encoders_and_paired_circuit_heads():
    routers = [JointRouter(64, method, 17) for method in METHODS]
    for router in routers[1:]:
        torch.testing.assert_close(router.input_proj.weight, routers[0].input_proj.weight, atol=0, rtol=0)
    torch.testing.assert_close(routers[2].output.weight, routers[3].output.weight, atol=0, rtol=0)


@pytest.mark.parametrize("qin,kin,kout,pos", [(4, 2, 3, (0, 2)), (5, 3, 2, (1, 3, 4)), (4, 2, 2, (2, 3))])
def test_projection_is_orthogonal_and_contraction_matches_dense(qin, kin, kout, pos):
    from fitmotn.experiments.qwen_joint.model import TensorExpert
    qout = qin - kin + kout
    w = torch.randn(2**qout, 2**qin, dtype=torch.float64)
    u = project_dense(w, qin, kin, kout, pos)
    dense = dense_from_tensor(u, qin, kin, kout, pos)
    torch.testing.assert_close(project_dense(dense, qin, kin, kout, pos), u)
    perturbation = dense_from_tensor(torch.randn_like(u), qin, kin, kout, pos)
    assert abs(float(((w - dense) * perturbation).sum())) < 1e-10
    x = torch.randn(3, 2**qin, dtype=torch.float64, requires_grad=True)
    expert = TensorExpert(u, pos, qin, kin, kout)
    torch.testing.assert_close(expert(x), x @ dense.T)
    assert torch.autograd.gradcheck(expert, (x,))


def small_projection(method="mlp"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(40)
        dense = torch.nn.Linear(64, 128, bias=False)
    return JointProjection(dense, method, AdapterConfig(free_input_sites=3, residual_rank=8, tail_rank=8), 3)


def test_real_sparse_dispatch_does_not_execute_unselected_experts():
    projection = small_projection()
    with torch.no_grad():
        projection.router.output.weight.zero_()
        projection.router.output.bias.copy_(torch.arange(16).float())
    calls = [0] * 16
    handles = []
    for eid, expert in enumerate(projection.experts):
        def hook(module, args, eid=eid):
            calls[eid] += len(args[0])
        handles.append(expert.register_forward_pre_hook(hook))
    projection(torch.randn(7, 64)).square().sum().backward()
    assert calls[:12] == [0] * 12 and calls[12:] == [7] * 4
    assert all(e.u.grad is None for e in projection.experts[:12])
    assert all(e.u.grad is not None and e.u.grad.norm() > 0 for e in projection.experts[12:])
    assert sum(calls) == 7 * 4
    for handle in handles:
        handle.remove()


def test_all_gates_share_distinct_expert_initialization():
    projections = [small_projection(m) for m in METHODS]
    assert len({p.conversion["expert_hash"] for p in projections}) == 1
    assert all(p.conversion["distinct_expert_hashes"] == 16 for p in projections)


def test_frozen_projection_reload_avoids_reconversion_and_preserves_output():
    original = small_projection("reencode_cz")
    reloaded = JointProjection.from_spec(64, 128, "reencode_cz",
        AdapterConfig(free_input_sites=3, residual_rank=8, tail_rank=8), 3, original.conversion)
    reloaded.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(9, 64)
    torch.testing.assert_close(original(x), reloaded(x), atol=0, rtol=0)


@pytest.mark.parametrize("method", (*METHODS, "dense"))
def test_real_qwen_forward_all_trainable_scopes_update(method):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(vocab_size=100, hidden_size=64, intermediate_size=192,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    model = Qwen3ForCausalLM(config)
    report = patch_qwen(model, method, AdapterConfig(free_input_sites=3, residual_rank=8, tail_rank=8))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    assert report["layers"] == [3]
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    frozen = tensor_hash({n: p for n, p in model.named_parameters() if not p.requires_grad})
    opt = MasterAdamW(model)
    x = torch.randint(0, 100, (2, 12))
    loss = model(x, labels=x, use_cache=False).loss + 0.01 * auxiliary_loss(model)
    loss.backward()
    assert torch.isfinite(loss)
    if method != "dense":
        assert any("router" in n and p.grad is not None and p.grad.norm() > 0 for n, p in model.named_parameters())
        assert any("experts" in n and p.grad is not None and p.grad.norm() > 0 for n, p in model.named_parameters())
    assert opt.step()["updated"]
    after = dict(model.named_parameters())
    assert any(not torch.equal(p, after[n]) for n, p in before.items())
    assert frozen == tensor_hash({n: p for n, p in model.named_parameters() if not p.requires_grad})
    for state in opt.optimizer.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32


def test_interrupted_training_replays_parameters_optimizer_rng_and_cursor(tmp_path):
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Dropout(0.2), torch.nn.Linear(7, 2))
    opt = MasterAdamW(model)
    sched = TokenSchedule(opt, 120)
    identity = {"model": "tiny", "data": "test", "code": "test"}

    def update(m, o, s):
        o.zero_grad()
        x = torch.randn(3, 5) * (random.random() + np.random.rand())
        s.set_next_update(12)
        m(x).square().sum().backward()
        o.step()
        s.advance(12)

    update(model, opt, sched)
    checkpoint = tmp_path / "step1"
    save_checkpoint(checkpoint, model, opt, sched, None, {"window": 3}, identity)
    update(model, opt, sched)
    update(model, opt, sched)
    expected = tensor_hash(model.state_dict())
    expected_optim = copy.deepcopy(opt.state_dict())
    expected_rng = torch.rand(4)
    restored = copy.deepcopy(model)
    opt2 = MasterAdamW(restored)
    sched2 = TokenSchedule(opt2, 120)
    cursor, _ = load_checkpoint(checkpoint, restored, opt2, sched2, None, identity)
    assert cursor == {"window": 3}
    update(restored, opt2, sched2)
    update(restored, opt2, sched2)
    assert expected == tensor_hash(restored.state_dict())
    for a, b in zip(expected_optim["masters"], opt2.state_dict()["masters"]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key in expected_optim["optimizer"]["state"]:
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(expected_optim["optimizer"]["state"][key][name],
                                       opt2.state_dict()["optimizer"]["state"][key][name], atol=0, rtol=0)
    torch.testing.assert_close(expected_rng, torch.rand(4), atol=0, rtol=0)
    assert sched2.tokens == sched.tokens == 36
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, model, opt, sched, None, cursor, identity)
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(checkpoint, restored, opt2, sched2, None, {"bad": True})


def test_exact_target_windows_ignore_boundaries_and_resume_partial_window(tmp_path):
    from fitmotn.experiments.qwen_joint.data import TargetWindows
    np.arange(3073, dtype="<u4").tofile(tmp_path / "train.tokens.u32")
    np.array([0, 8, 1024, 2048], dtype="<u8").tofile(tmp_path / "train.starts.u64")
    np.array([0, 1, 2], dtype="<u4").tofile(tmp_path / "train.seed17.order.u32")
    data = TargetWindows(tmp_path, "train", 17)
    ids, labels, meta = data.next(10)
    assert ids.tolist() == list(range(1024))
    assert meta["valid_targets"] == 10
    assert labels[7] == -100
    resumed = TargetWindows(tmp_path, "train", 17, data.state_dict())
    _, second, meta2 = resumed.next(32768)
    targets = labels[labels != -100].tolist() + second[second != -100].tolist()
    assert targets == [i for i in range(1, 1025) if i not in (8, 1024)]
    assert meta2["valid_targets"] == 1012
    assert resumed.state_dict() == {"position": 1, "offset": 0}


def test_connected_duplicate_groups_and_deterministic_split():
    from fitmotn.experiments.qwen_joint.data import Components, group_split
    groups = Components()
    groups.union(["id:a", "near:x"])
    groups.union(["id:b", "near:x", "content:y"])
    groups.union(["id:c", "content:y"])
    # A closed-text group can precede final cross-source family consolidation.
    groups.union(["id:d", "final-family:z"])
    groups.union(["id:d", "near:x"])
    assert groups.find("id:a") == groups.find("id:c")
    assert groups.find("id:a") == groups.find("id:d")
    assert group_split(groups.find("id:a")) == group_split(groups.find("id:c"))


def test_approved_content_hash_matches_predecessor_whitespace_definition():
    import hashlib
    from fitmotn.experiments.qwen_joint.data import approved_content_hash
    text = " Article\n\nbody\twith\u00a0spaces. "
    assert approved_content_hash(text) == hashlib.sha256(b"Article body with spaces.").hexdigest()
    assert approved_content_hash(text) != hashlib.sha256(text.encode("utf8")).hexdigest()


def test_chunked_loss_matches_qwen_shift_and_gradients():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from fitmotn.experiments.qwen_joint.task import next_target_loss
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=100, hidden_size=64, intermediate_size=192,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16)).eval()
    tokens = torch.randint(0, 100, (2, 13))
    expected = model(tokens, labels=tokens).loss
    expected.backward()
    grad = model.model.layers[-1].mlp.up_proj.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    ce, valid, _ = next_target_loss(model, tokens[:, :-1], tokens[:, 1:], logit_chunk=5)
    assert valid == 24
    torch.testing.assert_close(ce / valid, expected)
    (ce / valid).backward()
    torch.testing.assert_close(model.model.layers[-1].mlp.up_proj.weight.grad, grad, atol=1e-7, rtol=1e-4)


def test_bfloat16_parameters_keep_fp32_masters_and_moments():
    model = torch.nn.Linear(5, 3).to(torch.bfloat16)
    optimizer = MasterAdamW(model)
    model(torch.randn(4, 5, dtype=torch.bfloat16)).float().square().sum().backward()
    assert optimizer.step()["updated"]
    assert all(p.dtype == torch.float32 for p in optimizer.masters)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert all(s["exp_avg"].dtype == s["exp_avg_sq"].dtype == torch.float32 for s in optimizer.optimizer.state.values())


def test_routed_qwen_checkpoint_interruption_replay(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from fitmotn.experiments.qwen_joint.task import next_target_loss

    def create():
        torch.manual_seed(17)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=100, hidden_size=64, intermediate_size=192,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16))
        patch_qwen(model, "reencode_cz", AdapterConfig(free_input_sites=3, residual_rank=8, tail_rank=8))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model

    def step(model, optimizer, schedule):
        optimizer.zero_grad()
        tokens = torch.randint(0, 100, (1, 13))
        ce, valid, aux = next_target_loss(model, tokens[:, :-1], tokens[:, 1:], logit_chunk=5)
        schedule.set_next_update(valid)
        (ce / valid + 0.01 * aux).backward()
        optimizer.step()
        schedule.advance(valid)

    model = create()
    optimizer = MasterAdamW(model)
    schedule = TokenSchedule(optimizer, 120)
    step(model, optimizer, schedule)
    identity = {"kind": "routed_qwen_test", "seed": 17}
    save_checkpoint(tmp_path / "qwen", model, optimizer, schedule, None, {"position": 1, "offset": 2}, identity)
    step(model, optimizer, schedule)
    expected = tensor_hash(model.state_dict())
    model2 = create()
    opt2 = MasterAdamW(model2)
    sched2 = TokenSchedule(opt2, 120)
    cursor, _ = load_checkpoint(tmp_path / "qwen", model2, opt2, sched2, None, identity)
    assert cursor == {"position": 1, "offset": 2}
    step(model2, opt2, sched2)
    assert tensor_hash(model2.state_dict()) == expected


def test_gpu_lease_exclusion_ownership_and_stop(monkeypatch, tmp_path):
    import json
    from fitmotn.experiments.qwen_joint import resources
    monkeypatch.setattr(resources, "gpu_snapshot", lambda uuid: {"idle": True, "uuid": uuid})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with resources.GPULease(tmp_path, resources.PRO4000, "test", tmp_path / "run") as first:
        with pytest.raises(FileExistsError):
            with resources.GPULease(tmp_path, resources.PRO4000, "second", tmp_path / "other"):
                pass
        assert first.path.exists()
    assert not first.path.exists()
    with resources.GPULease(tmp_path, resources.PRO4000, "test", tmp_path / "run") as lease:
        lease.path.write_text(json.dumps({"pid": 999999, "token": "different-owner"}), encoding="utf8")
    assert lease.path.exists()
    (tmp_path / "STOP").touch()
    with pytest.raises(RuntimeError, match="Stop marker"):
        with resources.GPULease(tmp_path, resources.V100[0], "test", tmp_path / "run"):
            pass


def test_wddm_filter_keeps_unknown_compute_processes_blocking():
    from fitmotn.experiments.qwen_joint.resources import desktop_process
    assert desktop_process("dwm.exe", "")
    assert desktop_process("Cortana.exe", r"C:\Program Files\WindowsApps\Microsoft.549981C3F5F10_4.2308.1005.0_x64__8wekyb3d8bbwe\Cortana.exe")
    assert desktop_process("explorer.exe", r"C:\Windows\explorer.exe")
    assert not desktop_process("python.exe", r"E:\AI\env\python.exe")
    assert not desktop_process("Cortana.exe", r"E:\unknown\Cortana.exe")
    assert not desktop_process("", "[Insufficient Permissions]")


@pytest.mark.parametrize("entry", ["acceptance", "train"])
def test_cli_fresh_output_refusal_preserves_existing_evidence(monkeypatch, tmp_path, entry):
    import importlib
    from fitmotn.experiments.qwen_joint.resources import PRO4000
    module = importlib.import_module("fitmotn.experiments.qwen_joint." + entry)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "RESULT.json").write_text('{"passed": true}', encoding="utf8")
    args = (["probe", "--gpu", PRO4000] if entry == "acceptance" else
            ["smoke", "--method", "dense", "--data", "unused", "--gpu", PRO4000])
    monkeypatch.setattr("sys.argv", [entry, *args, "--output", str(output)])
    with pytest.raises(FileExistsError):
        module.main()
    assert list(output.iterdir()) == [output / "RESULT.json"]
    assert (output / "RESULT.json").read_text(encoding="utf8") == '{"passed": true}'
