from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from fitmotn.checkpointing import save_fitmotn_metadata
from fitmotn.config.defaults import make_default_config
from fitmotn.eval import restore as restore_module
from fitmotn.eval.restore import restore_fitmotn_model
from fitmotn.init.approx import _configure_trainable_subset
from fitmotn.model import SparseMiXTFFNLayer
from fitmotn.patching import build_patch_model_config, patch_qwen_ffn_layers, set_trainable_patch_only
from fitmotn.rl.vllm_weight_mapping import collect_module_aware_motn_keys
from fitmotn.sparse_mixt import BlockPartitionedLinear, SparseMiXTLinear, largest_power_of_d_leq
from fitmotn.sparse_mixt import LowRankLinear


def _cfg(**overrides):
    cfg = {
        "E": 1,
        "d": 2,
        "k_in": 2,
        "topk": 1,
        "gate_type": "topk",
        "capacity_factor": 100.0,
        "min_capacity": 1,
        "drop_tokens": False,
        "pos_strategy": "sliding",
        "global_expert_enabled": False,
        "dtype": torch.float32,
    }
    cfg.update(overrides)
    return cfg


def test_largest_power_split():
    assert largest_power_of_d_leq(2560, 2) == 2048
    assert largest_power_of_d_leq(9216, 2) == 8192
    assert largest_power_of_d_leq(81, 3) == 81


def test_stage1_dense_four_block_is_exact_and_trace_is_auditable():
    torch.manual_seed(1)
    dense = nn.Linear(6, 10, bias=False)
    block = BlockPartitionedLinear(6, 10, 4, 8)
    block.set_from_dense_weight(dense.weight)
    x = torch.randn(2, 3, 6)

    lines = []
    trace = lambda message: (lines.append(message), print(message))[-1]
    actual = block.forward_with_trace(x, trace=trace)
    expected = dense(x)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    trace_text = "\n".join(lines)
    assert "y0=W00(x0)+W01(x1)" in trace_text
    assert "padding=False crop=False" in trace_text


def test_stage2_sparse_mixt_projection_exact_small_network_and_trace():
    torch.manual_seed(2)
    dense = nn.Linear(6, 10, bias=False)
    sparse = SparseMiXTLinear(
        6,
        10,
        cfg=_cfg(),
        ranks={"01": 2, "10": 2, "11": 2},
        main_in_features=4,
        main_out_features=8,
    )
    sparse.initialize_from_dense_weight(dense.weight, initialize_main_exact=True)
    x = torch.randn(7, 6)

    lines = []
    trace = lambda message: (lines.append(message), print(message))[-1]
    actual, _ = sparse.forward_with_trace(x, trace=trace)
    expected = dense(x)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    trace_text = "\n".join(lines)
    assert "q_in=2 q_out=3 k_in=2 k_out=3" in trace_text
    assert "experts=1" in trace_text
    assert "padding=False crop=False" in trace_text


def test_sparse_mixt_full_real_router_reads_real_not_padded_dimension():
    sparse = SparseMiXTLinear(
        6,
        10,
        cfg=_cfg(),
        ranks={"01": 2, "10": 2, "11": 2},
        main_in_features=4,
        main_out_features=8,
        router_input_policy="full_real",
    )
    assert sparse.core.gate.router.data_dim == 6
    y, _ = sparse(torch.randn(3, 6))
    assert y.shape == (3, 10)


def test_sparse_mixt_backward_reaches_main_router_and_all_boundaries():
    sparse = SparseMiXTLinear(
        6,
        10,
        cfg=_cfg(E=2, topk=2, k_in=1),
        ranks={"01": 2, "10": 2, "11": 2},
        main_in_features=4,
        main_out_features=8,
    )
    x = torch.randn(5, 6, requires_grad=True)
    target = torch.randn(5, 10)
    optimizer = torch.optim.SGD(sparse.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        y, _ = sparse(x)
        (y - target).square().mean().backward()
        optimizer.step()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(block.U.grad is not None and block.U.grad.abs().sum() > 0 for block in sparse.core.blocks)
    router_grad = sparse.core.gate.router.linear.weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0
    for branch in (sparse.lr01, sparse.lr10, sparse.lr11):
        assert branch.A.weight.grad is not None and branch.A.weight.grad.abs().sum() > 0
        assert branch.B.weight.grad is not None and branch.B.weight.grad.abs().sum() > 0


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(6, 10, bias=False)
        self.up_proj = nn.Linear(6, 10, bias=False)
        self.down_proj = nn.Linear(10, 6, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _sparse_ffn_cfg():
    return {
        **_cfg(),
        "patch_backend": "sparse_mixt",
        "sparse_mixt": {
            "hidden_main": 4,
            "intermediate_main": 8,
            "router_input_policy": "main",
            "gate_ranks": {"01": 2, "10": 2, "11": 2},
            "up_ranks": {"01": 2, "10": 2, "11": 2},
            "down_ranks": {"01": 2, "10": 2, "11": 2},
        },
    }


def test_stage3_full_ffn_replacement_is_exact_when_all_blocks_are_exact():
    torch.manual_seed(3)
    dense = _TinyMLP()
    sparse = SparseMiXTFFNLayer(dense, _sparse_ffn_cfg(), device=torch.device("cpu"))
    for name in ("gate_proj", "up_proj", "down_proj"):
        getattr(sparse, name).initialize_from_dense_weight(getattr(dense, name).weight, initialize_main_exact=True)
    x = torch.randn(2, 4, 6)

    expected = dense(x)
    lines = []
    trace = lambda message: (lines.append(message), print(message))[-1]
    actual = sparse.forward_with_trace(x, trace=trace)

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    trace_text = "\n".join(lines)
    assert trace_text.count("padding=False crop=False") == 3
    assert "h=act(gate) * up" in trace_text
    assert "q_in=3 q_out=2 k_in=3 k_out=2" in trace_text


def test_patching_selects_new_backend_and_preserves_existing_projection_interfaces():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _TinyMLP()

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    decoder = Decoder()
    patch_qwen_ffn_layers(decoder, [0], _sparse_ffn_cfg(), torch.device("cpu"))
    patched = decoder.layers[0].mlp
    assert isinstance(patched, SparseMiXTFFNLayer)
    assert patched.patch_backend == "sparse_mixt"
    assert patched.gate_proj.core.q_in == 2
    assert patched.down_proj.core.k_in == patched.gate_proj.core.k_out
    assert patched(torch.randn(2, 6)).shape == (2, 6)


def test_build_patch_config_carries_versioned_sparse_layout():
    cfg = make_default_config()
    cfg.model.patch_backend = "sparse_mixt"
    cfg.model.sparse_mixt["hidden_main"] = 2048
    patch_cfg = build_patch_model_config(cfg)
    assert patch_cfg["patch_backend"] == "sparse_mixt"
    assert patch_cfg["sparse_mixt"]["backend_version"] == 1
    assert patch_cfg["sparse_mixt"]["hidden_main"] == 2048


def test_approx_calibration_includes_all_low_rank_boundary_parameters():
    sparse = SparseMiXTLinear(
        6,
        10,
        cfg=_cfg(),
        ranks={"01": 2, "10": 2, "11": 2},
        main_in_features=4,
        main_out_features=8,
    )
    params = _configure_trainable_subset(sparse, [0])
    param_ids = {id(param) for param in params}
    for branch in (sparse.lr01, sparse.lr10, sparse.lr11):
        assert all(id(param) in param_ids for param in branch.parameters())


def test_vllm_patch_mapping_includes_sparse_boundary_factors():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _TinyMLP()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    model = Model()
    patch_qwen_ffn_layers(model, [0], _sparse_ffn_cfg(), torch.device("cpu"))
    set_trainable_patch_only(model)
    roles = collect_module_aware_motn_keys(model)
    assert any("lr01.A.weight" in name for name in roles)
    assert any("lr10.B.weight" in name for name in roles)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable <= set(roles)


def test_sparse_checkpoint_roundtrip_reconstructs_backend_and_all_weights(tmp_path, monkeypatch):
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([type("Layer", (nn.Module,), {"__init__": lambda self: (nn.Module.__init__(self), setattr(self, "mlp", _TinyMLP()))[-1]})()])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    source = Model()
    patch_cfg = _sparse_ffn_cfg()
    patch_qwen_ffn_layers(source, [0], patch_cfg, torch.device("cpu"))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_fitmotn_metadata(
        checkpoint,
        {
            "base_model_path": "fake-base",
            "layers_to_patch": [0],
            "patch_backend": "sparse_mixt",
            "patch_cfg": patch_cfg,
            "motn_cfg": patch_cfg,
            "patch_state_dict": source.state_dict(),
        },
    )

    monkeypatch.setattr(
        restore_module,
        "load_causal_lm_and_tokenizer",
        lambda *args, **kwargs: (Model(), object(), torch.float32),
    )
    restored, _, metadata = restore_fitmotn_model(checkpoint, device="cpu")
    assert metadata["patch_backend"] == "sparse_mixt"
    assert isinstance(restored.model.layers[0].mlp, SparseMiXTFFNLayer)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_one_sided_tail_projection_uses_only_mathematically_required_branches():
    sparse = SparseMiXTLinear(
        8,
        12,
        cfg=_cfg(k_in=3),
        ranks={"01": 2, "10": 2, "11": 2},
        main_in_features=8,
        main_out_features=8,
    )
    assert sparse.lr01 is None
    assert sparse.lr10 is not None
    assert sparse.lr11 is None
    lines = []
    y, _ = sparse.forward_with_trace(torch.randn(3, 8), trace=lines.append)
    assert y.shape == (3, 12)
    assert sparse.in_pad == 8 and sparse.out_pad == 12
    assert "LR01" not in "\n".join(lines) and "LR11" not in "\n".join(lines)


def test_unknown_sparse_backend_version_fails_closed():
    cfg = _cfg(sparse_mixt={"backend_version": 99})
    with pytest.raises(ValueError, match="backend_version"):
        SparseMiXTLinear(
            6,
            10,
            cfg=cfg,
            ranks={"01": 2, "10": 2, "11": 2},
            main_in_features=4,
            main_out_features=8,
        )


def test_qwen35_4b_layout_derives_expected_q_and_k_without_allocating_padding():
    cfg = _cfg(E=1, topk=1, k_in=8)
    gate = SparseMiXTLinear(
        2560,
        9216,
        cfg=cfg,
        ranks={"01": 1, "10": 1, "11": 1},
    )
    down_cfg = dict(cfg)
    down_cfg["k_in"] = gate.core.k_out
    down = SparseMiXTLinear(
        9216,
        2560,
        cfg=down_cfg,
        ranks={"01": 1, "10": 1, "11": 1},
    )
    assert (gate.main_in_features, gate.main_out_features) == (2048, 8192)
    assert (gate.core.q_in, gate.core.q_out, gate.core.k_in, gate.core.k_out) == (11, 13, 8, 10)
    assert (down.core.q_in, down.core.q_out, down.core.k_in, down.core.k_out) == (13, 11, 10, 8)
    assert gate.in_pad == 2560 and gate.out_pad == 9216
    assert down.in_pad == 9216 and down.out_pad == 2560


def test_soft_routing_and_global_expert_are_inherited_from_existing_core():
    sparse = SparseMiXTLinear(
        6,
        10,
        cfg=_cfg(E=2, k_in=1, gate_type="softmax", global_expert_enabled=True),
        ranks={"01": 2, "10": 2, "11": 2},
        main_in_features=4,
        main_out_features=8,
    )
    x = torch.randn(4, 6)
    with_global, _ = sparse(x)
    without_global, _ = sparse(x, use_global_expert=False)
    assert sparse.core.global_block is not None
    assert sparse.core.gate.__class__.__name__ == "SoftGate"
    assert not torch.equal(with_global, without_global)


def test_sparse_ffn_warmup_scaling_flattens_tokens_and_preserves_boundary_path():
    cfg = _sparse_ffn_cfg()
    cfg.update(E=2, topk=2, k_in=1)
    layer = SparseMiXTFFNLayer(_TinyMLP(), cfg, device=torch.device("cpu"))
    layer.train()
    layout = {"slide_indices": [0], "random_indices": [1]}
    layer.fitmotn_block_layout = {name: layout for name in ("gate_proj", "up_proj", "down_proj")}
    layer.fitmotn_expert_warmup_state = {
        "enabled": True,
        **{name: {"random_scale": 0.25} for name in ("gate_proj", "up_proj", "down_proj")},
    }
    captured = {}
    original = layer.gate_proj.core.forward

    def wrapped(x, *args, **kwargs):
        captured["shape"] = tuple(x.shape)
        captured["probs"] = kwargs.get("probs")
        return original(x, *args, **kwargs)

    layer.gate_proj.core.forward = wrapped
    y = layer(torch.randn(2, 3, 6))
    assert y.shape == (2, 3, 6)
    assert captured["shape"] == (6, 4)
    assert captured["probs"].shape == (6, 2)
    torch.testing.assert_close(captured["probs"].sum(dim=-1), torch.ones(6))


def test_sparse_ffn_bfloat16_smoke_preserves_public_dtype():
    cfg = _sparse_ffn_cfg()
    cfg["dtype"] = torch.bfloat16
    layer = SparseMiXTFFNLayer(_TinyMLP(), cfg, device=torch.device("cpu"), dtype=torch.bfloat16)
    x = torch.randn(2, 6, dtype=torch.bfloat16)
    y = layer(x)
    assert y.dtype == torch.bfloat16
    assert y.shape == x.shape


def test_randomized_svd_boundary_initialization_is_finite_and_improves_zero_baseline():
    torch.manual_seed(7)
    weight = torch.randn(12, 9)
    branch = LowRankLinear(9, 12, rank=3)
    report = branch.set_from_dense_weight(weight, method="randomized_svd", oversampling=3, niter=2)
    reconstruction = branch.A.weight @ branch.B.weight
    zero_error = weight.norm()
    fitted_error = (weight - reconstruction).norm()
    assert report["method"] == "randomized_svd"
    assert torch.isfinite(reconstruction).all()
    assert fitted_error < zero_error
