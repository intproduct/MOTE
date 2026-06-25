from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg
if "MOTE.config" not in sys.modules:
    config_pkg = types.ModuleType("MOTE.config")
    config_pkg.__path__ = [str(ROOT / "config")]
    sys.modules["MOTE.config"] = config_pkg
if "MOTE.train" not in sys.modules:
    train_pkg = types.ModuleType("MOTE.train")
    train_pkg.__path__ = [str(ROOT / "train")]
    sys.modules["MOTE.train"] = train_pkg

from MOTE.config.loader import load_config_from_json
from MOTE.gate import GateConfig, SoftGate, TopKGate, gate_factory_config, normalize_legacy_gate_state_dict_for_model, topk_routing_hard
from MOTE.patching import build_patch_model_config, patch_qwen_ffn_layers


PORTABLE_PATHS = {
    "model": {"model_path": "models/Qwen3-0.6B"},
    "data": {
        "tok_shard_dir": "data/wiki24_tok",
        "fineweb_cache_path": "cache/fineweb_sample10bt",
        "code_cache_path": "cache/the_stack_v2",
        "gsm8k_cache_path": "cache/gsm8k_main",
        "math_cache_root": "cache/hendrycks_math",
    },
    "output": {"root_dir": "runs"},
}


class FakeQwenMLP(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()


class FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeQwenMLP()


class FakeInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeDecoderLayer()])


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeInnerModel()


class GateRouterArchTests(unittest.TestCase):
    def test_old_json_defaults_to_linear_and_fc_is_accessible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            payload = json.loads(json.dumps(PORTABLE_PATHS))
            payload["train"] = {"lr_scheduler_type": "constant"}
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config_from_json(path)

        self.assertEqual(cfg.model.gate_arch, "linear")
        gate = TopKGate(data_dim=8, num_experts=4, k=2, min_capacity=1)
        x = torch.randn(7, 8)
        probs, mask, aux = gate(x)
        self.assertEqual(tuple(probs.shape), (7, 4))
        self.assertEqual(tuple(mask.shape), (7, 4))
        self.assertIn("l_aux", aux)
        self.assertIsNotNone(gate.fc)
        self.assertEqual(tuple(gate.fc.weight.shape), (4, 8))

    def test_legacy_fc_weight_checkpoint_key_maps_to_router_linear(self):
        model = nn.Module()
        model.core = nn.Module()
        model.core.gate = TopKGate(data_dim=8, num_experts=4, k=2, min_capacity=1)
        state = model.state_dict()
        old_state = {}
        for key, value in state.items():
            old_key = key.replace("core.gate.router.linear.weight", "core.gate.fc.weight")
            old_state[old_key] = value.clone()

        normalized = normalize_legacy_gate_state_dict_for_model(model, old_state)
        self.assertIn("core.gate.router.linear.weight", normalized)
        self.assertNotIn("core.gate.fc.weight", normalized)
        model.load_state_dict(normalized, strict=True)

        gate = TopKGate(data_dim=8, num_experts=4, k=2, min_capacity=1)
        gate_state = normalize_legacy_gate_state_dict_for_model(gate, {"fc.weight": gate.fc.weight.detach().clone()})
        self.assertIn("router.linear.weight", gate_state)
        self.assertNotIn("fc.weight", gate_state)
        gate.load_state_dict(gate_state, strict=True)

    def test_mlp_topk_forward_aux_and_usage_tracking(self):
        gate = gate_factory_config(
            GateConfig(
                gate_type="topk",
                data_dim=8,
                num_experts=4,
                k=2,
                min_capacity=1,
                gate_arch="mlp",
                gate_hidden_dim=6,
                gate_activation="silu",
                gate_norm="layernorm",
            )
        )
        x = torch.randn(9, 8)
        probs, mask, aux = gate(x)
        self.assertEqual(tuple(probs.shape), (9, 4))
        self.assertEqual(tuple(mask.shape), (9, 4))
        self.assertEqual(tuple(aux["importance"].shape), (4,))
        self.assertIsNotNone(gate.runtime_expert_counts)
        self.assertIsNone(gate.fc)

        soft_gate = SoftGate(data_dim=8, num_experts=4, gate_arch="mlp", gate_hidden_dim=6)
        soft_probs, soft_mask, soft_aux = soft_gate(x)
        self.assertEqual(tuple(soft_probs.shape), (9, 4))
        self.assertIsNone(soft_mask)
        self.assertIn("l_aux", soft_aux)

    def test_topk_routing_hard_matches_argsort_selected_set_without_ties(self):
        probs = torch.tensor(
            [
                [[0.11, 0.23, 0.07, 0.41, 0.18], [0.32, 0.04, 0.29, 0.21, 0.14]],
                [[0.09, 0.51, 0.13, 0.19, 0.08], [0.27, 0.31, 0.05, 0.12, 0.25]],
            ],
            dtype=torch.float32,
        )
        for k in (1, 3):
            _, mask = topk_routing_hard(probs, k)
            flat = probs.reshape(-1, probs.shape[-1])
            idx = torch.argsort(flat, dim=-1)[:, -min(k, probs.shape[-1]) :]
            baseline = torch.zeros_like(flat).scatter(1, idx, 1.0).reshape_as(probs)
            # Tied top-k indices are not stable across selection algorithms; this
            # no-tie fixture checks selected sets/masks rather than index order.
            self.assertTrue(torch.equal(mask, baseline))

    def test_topk_routing_hard_rejects_zero_k(self):
        with self.assertRaisesRegex(ValueError, "at least one expert"):
            topk_routing_hard(torch.rand(2, 4), 0)

    def test_residual_mlp_scale_zero_equals_base_linear(self):
        gate = TopKGate(
            data_dim=8,
            num_experts=4,
            k=2,
            min_capacity=1,
            gate_arch="residual_mlp",
            gate_hidden_dim=6,
            gate_residual_delta_scale=0.0,
        )
        x = torch.randn(5, 8)
        self.assertTrue(any("router.base_linear" in name for name, _ in gate.named_parameters()))
        self.assertTrue(any("router.mlp" in name for name, _ in gate.named_parameters()))
        self.assertTrue(torch.allclose(gate.router(x), gate.router.base_linear(x), atol=1e-7))
        probs, mask, _ = gate(x)
        self.assertEqual(tuple(probs.shape), (5, 4))
        self.assertEqual(tuple(mask.shape), (5, 4))

    def test_adtn_fixed_ignores_gate_arch_and_does_not_create_gate(self):
        cfg = types.SimpleNamespace()
        cfg.model = types.SimpleNamespace(
            patch_backend="adtn_fixed",
            E=4,
            d=2,
            k_in=2,
            pos_strategy="warmup",
            init_gamma=0.6,
            warmup_ratio=0.5,
            warmup_stride=1,
            gate_arch="mlp",
        )
        cfg.approx_init = types.SimpleNamespace(enabled=False, subset_mode="")
        patch_cfg = build_patch_model_config(cfg)
        self.assertNotIn("gate_arch", patch_cfg)
        model = patch_qwen_ffn_layers(FakeModel(), [0], patch_cfg, device=torch.device("cpu"), dtype=torch.float32)
        layer = model.model.layers[0].mlp
        self.assertFalse(hasattr(layer.gate_proj.core, "gate"))
        self.assertFalse(hasattr(layer.up_proj.core, "gate"))
        self.assertFalse(hasattr(layer.down_proj.core, "gate"))


if __name__ == "__main__":
    unittest.main()
