from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from fitmotn.checkpointing import save_fitmotn_metadata
from fitmotn.config.defaults import make_default_config
from fitmotn.eval import restore as restore_module
from fitmotn.eval.restore import restore_fitmotn_model
from fitmotn.init.approx import _configure_trainable_subset
from fitmotn.init.approx import collect_dense_ffn_targets
from fitmotn.mixed_mixt import MixedMiXTLinear
from fitmotn.model import MixedMiXTFFNLayer
from fitmotn.patching import (
    build_patch_model_config,
    patch_qwen_ffn_layers,
    set_motn_gate_trainable,
    set_motn_temperature,
    set_motn_usage_tracking,
    set_trainable_patch_only,
    summarize_motn_gate_routers,
)
from fitmotn.rl.vllm_weight_mapping import collect_module_aware_motn_keys


def _base_cfg(**overrides):
    cfg = {
        "patch_backend": "mixed_mixt",
        "E": 1,
        "d": 2,
        "k_in": 2,
        "topk": 1,
        "gate_type": "topk",
        "capacity_factor": 100.0,
        "min_capacity": 1,
        "drop_tokens": False,
        "pos_strategy": "random",
        "global_expert_enabled": False,
        "dtype": torch.float32,
        "mixed_mixt": {"backend_version": 1},
    }
    cfg.update(overrides)
    return cfg


def _full_bonds_6_to_10():
    return {"m00": 2, "m01": 1, "m10": 2, "m11": 1}


def _full_bonds_10_to_6():
    return {"m00": 3, "m01": 1, "m10": 3, "m11": 1}


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(6, 10, bias=False)
        self.up_proj = nn.Linear(6, 10, bias=False)
        self.down_proj = nn.Linear(10, 6, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _exact_ffn_cfg():
    return {
        **_base_cfg(),
        "mixed_mixt": {
            "backend_version": 1,
            "hidden_main": 4,
            "intermediate_main": 8,
            "router_input_policy": "full_real",
            "dense_init": "full_site_exact",
            "gate_bonds": _full_bonds_6_to_10(),
            "up_bonds": _full_bonds_6_to_10(),
            "down_bonds": _full_bonds_10_to_6(),
        },
    }


def test_four_full_site_mixt_quadrants_exactly_reproduce_dense_and_trace():
    torch.manual_seed(11)
    dense = nn.Linear(6, 10, bias=False)
    mixed = MixedMiXTLinear(
        6,
        10,
        cfg=_base_cfg(),
        bonds=_full_bonds_6_to_10(),
        main_in_features=4,
        main_out_features=8,
    )
    report = mixed.initialize_from_dense_weight_exact(dense.weight)
    x = torch.randn(2, 3, 6)
    lines = []
    actual, _ = mixed.forward_with_trace(x, trace=lambda line: (lines.append(line), print(line))[-1])
    expected = dense(x)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert set(report) == {"m00", "m01", "m10", "m11"}
    text = "\n".join(lines)
    assert "router_input=full_real calls=1" in text
    assert "y0=M00(x0)+M01(x1)" in text
    assert "padding=False crop=False" in text


def test_shared_gate_is_called_once_for_all_four_quadrants():
    mixed = MixedMiXTLinear(
        6,
        10,
        cfg=_base_cfg(),
        bonds=_full_bonds_6_to_10(),
        main_in_features=4,
        main_out_features=8,
    )
    calls = []
    handle = mixed.core.gate.register_forward_hook(lambda *args: calls.append(1))
    try:
        y, _ = mixed(torch.randn(4, 6))
    finally:
        handle.remove()
    assert y.shape == (4, 10)
    assert len(calls) == 1
    assert all(quadrant.gate is None for quadrant in mixed.core.quadrants.values())


def test_local_bond_training_reaches_router_and_every_quadrant_expert():
    cfg = _base_cfg(E=2, topk=2, k_in=2, global_expert_enabled=True)
    bonds = {"m00": 2, "m01": 1, "m10": 2, "m11": 1}
    mixed = MixedMiXTLinear(
        12,
        20,
        cfg=cfg,
        bonds=bonds,
        main_in_features=8,
        main_out_features=16,
    )
    optimizer = torch.optim.AdamW(mixed.parameters(), lr=0.03)
    x = torch.randn(7, 12)
    target = torch.randn(7, 20)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        y, _ = mixed(x)
        loss = (y - target).square().mean()
        loss.backward()
        optimizer.step()

    router_grad = mixed.core.gate.router.linear.weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0
    for quadrant in mixed.core.quadrants.values():
        assert quadrant.global_block is not None
        assert all(block.U.grad is not None and block.U.grad.abs().sum() > 0 for block in quadrant.blocks)
        assert quadrant.global_block.U.grad is not None and quadrant.global_block.U.grad.abs().sum() > 0


def test_full_qwen_ffn_replacement_exact_with_four_mixt_groups_per_projection():
    torch.manual_seed(12)
    dense = _TinyMLP()
    reference = _TinyMLP()
    reference.load_state_dict(dense.state_dict())
    mixed = MixedMiXTFFNLayer(dense, _exact_ffn_cfg(), device=torch.device("cpu"))
    x = torch.randn(2, 4, 6)
    expected = reference(x)
    lines = []
    actual = mixed.forward_with_trace(x, trace=lambda line: (lines.append(line), print(line))[-1])

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    text = "\n".join(lines)
    assert text.count("shared routing") == 3
    assert text.count("padding=False crop=False") == 3
    assert "h=act(gate) * up" in text


def test_patcher_selects_mixed_backend_and_build_config_carries_layout():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _TinyMLP()

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    decoder = Decoder()
    patch_qwen_ffn_layers(decoder, [0], _exact_ffn_cfg(), torch.device("cpu"))
    assert isinstance(decoder.layers[0].mlp, MixedMiXTFFNLayer)
    assert decoder.layers[0].mlp.patch_backend == "mixed_mixt"

    cfg = make_default_config()
    cfg.model.patch_backend = "mixed_mixt"
    cfg.model.mixed_mixt["hidden_main"] = 2048
    patch_cfg = build_patch_model_config(cfg)
    assert patch_cfg["patch_backend"] == "mixed_mixt"
    assert patch_cfg["mixed_mixt"]["backend_version"] == 1
    assert patch_cfg["mixed_mixt"]["hidden_main"] == 2048


def test_qwen35_nested_language_model_layer_path_is_patchable():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _TinyMLP()

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LanguageModel()

    class ConditionalGeneration(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()

    model = ConditionalGeneration()
    targets = collect_dense_ffn_targets(model, [0])
    assert targets[0]["gate_proj"].shape == (10, 6)
    patch_qwen_ffn_layers(model, [0], _exact_ffn_cfg(), torch.device("cpu"))
    assert isinstance(model.model.language_model.layers[0].mlp, MixedMiXTFFNLayer)


def test_shared_router_controls_and_usage_report_are_inherited():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _TinyMLP()

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    decoder = Decoder()
    patch_qwen_ffn_layers(decoder, [0], _exact_ffn_cfg(), torch.device("cpu"))
    mixed = decoder.layers[0].mlp
    router_names = [name for name, _ in decoder.named_modules() if name.endswith("core.gate.router")]
    assert len(router_names) == 3
    assert not any(
        any(f".quadrants.{quadrant}.gate" in name for quadrant in ("m00", "m01", "m10", "m11"))
        for name, _ in decoder.named_modules()
    )

    set_motn_temperature(decoder, 0.75)
    assert all(getattr(getattr(mixed, name).core.gate, "temperature") == 0.75 for name in ("gate_proj", "up_proj", "down_proj"))
    set_motn_gate_trainable(decoder, False)
    assert all(not param.requires_grad for name in ("gate_proj", "up_proj", "down_proj") for param in getattr(mixed, name).core.gate.parameters())
    set_motn_usage_tracking(decoder, True)
    mixed(torch.randn(2, 3, 6))
    report = mixed.gate_proj.core.materialize_usage_report()
    assert report["shared_router"] is True
    assert report["quadrants"] == ["m00", "m01", "m10", "m11"]

    summary = summarize_motn_gate_routers(decoder, "mixed_mixt")
    assert summary["router_param_count"] > 0
    assert summary["router_param_ratio_vs_blocks"] is not None


def test_approx_and_vllm_include_every_mixed_quadrant():
    mixed = MixedMiXTLinear(
        12,
        20,
        cfg=_base_cfg(E=2, topk=2),
        bonds={"m00": 2, "m01": 1, "m10": 2, "m11": 1},
        main_in_features=8,
        main_out_features=16,
    )
    params = _configure_trainable_subset(mixed, [0])
    ids = {id(param) for param in params}
    for quadrant in mixed.core.quadrants.values():
        assert id(quadrant.blocks[0].U) in ids
        assert id(quadrant.blocks[1].U) not in ids

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _TinyMLP()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    model = Model()
    patch_qwen_ffn_layers(model, [0], _exact_ffn_cfg(), torch.device("cpu"))
    set_trainable_patch_only(model)
    roles = collect_module_aware_motn_keys(model)
    assert any("core.quadrants.m00.blocks.0.U" in name for name in roles)
    assert any("core.quadrants.m11.blocks.0.U" in name for name in roles)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable <= set(roles)


def test_mixed_checkpoint_roundtrip_is_strict(tmp_path, monkeypatch):
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([type("Layer", (nn.Module,), {"__init__": lambda self: (nn.Module.__init__(self), setattr(self, "mlp", _TinyMLP()))[-1]})()])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    source = Model()
    cfg = _exact_ffn_cfg()
    patch_qwen_ffn_layers(source, [0], cfg, torch.device("cpu"))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_fitmotn_metadata(
        checkpoint,
        {
            "base_model_path": "fake-base",
            "layers_to_patch": [0],
            "patch_backend": "mixed_mixt",
            "patch_cfg": cfg,
            "motn_cfg": cfg,
            "patch_state_dict": source.state_dict(),
        },
    )
    monkeypatch.setattr(
        restore_module,
        "load_causal_lm_and_tokenizer",
        lambda *args, **kwargs: (Model(), object(), torch.float32),
    )
    restored, _, metadata = restore_fitmotn_model(checkpoint, device="cpu")
    assert metadata["patch_backend"] == "mixed_mixt"
    assert isinstance(restored.model.layers[0].mlp, MixedMiXTFFNLayer)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_qwen35_layout_and_bonds_are_power_of_two_without_padding():
    cfg = _base_cfg(E=1, topk=1)
    mixed = MixedMiXTLinear(
        2560,
        9216,
        cfg=cfg,
        bonds={"m00": 8, "m01": 5, "m10": 6, "m11": 5},
    )
    expected = {
        "m00": (2048, 8192, 11, 13, 8, 10),
        "m01": (512, 8192, 9, 13, 5, 9),
        "m10": (2048, 1024, 11, 10, 6, 5),
        "m11": (512, 1024, 9, 10, 5, 6),
    }
    for name, values in expected.items():
        q = mixed.core.quadrants[name]
        assert (q.dim_input, q.dim_output, q.q_in, q.q_out, q.k_in, q.k_out) == values
    assert mixed.in_pad == 2560 and mixed.out_pad == 9216


def test_mixed_backend_version_fails_closed():
    cfg = _base_cfg(mixed_mixt={"backend_version": 99})
    with pytest.raises(ValueError, match="backend_version"):
        MixedMiXTLinear(
            6,
            10,
            cfg=cfg,
            bonds=_full_bonds_6_to_10(),
            main_in_features=4,
            main_out_features=8,
        )
