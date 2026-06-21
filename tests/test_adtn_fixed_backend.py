from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
import warnings
from collections import deque
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
if "MOTE.eval" not in sys.modules:
    eval_pkg = types.ModuleType("MOTE.eval")
    eval_pkg.__path__ = [str(ROOT / "eval")]
    sys.modules["MOTE.eval"] = eval_pkg

from MOTE.baselines.adtn_fixed import ADTNBaselineFFNLayer
from MOTE.baselines import adtn_fixed as adtn_fixed_module
from MOTE.checkpointing import save_fitmotn_metadata
from MOTE.config.defaults import make_default_config
from MOTE.config.loader import _finalize_train_config, load_config_from_json
from MOTE.eval import restore as restore_module
from MOTE.eval.restore import restore_fitmotn_model
from MOTE.init.approx import fit_patched_ffn_layer
from MOTE import model as motn_model_module
from MOTE.patching import build_patch_model_config, patch_qwen_ffn_layers, set_trainable_patch_only
from MOTE.train.observability import build_checkpoint_metadata, build_run_summary


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


def with_portable_paths(payload):
    merged = json.loads(json.dumps(PORTABLE_PATHS))
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


class FakeQwenMLP(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.mlp = FakeQwenMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)


class FakeInnerModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([FakeDecoderLayer(hidden_size, intermediate_size) for _ in range(n_layers)])


class FakeModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.model = FakeInnerModel(n_layers=n_layers, hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.config = types.SimpleNamespace(num_hidden_layers=n_layers)


class ADTNFixedBackendTests(unittest.TestCase):
    def _cfg(self, *, patch_backend: str = "motn"):
        cfg = make_default_config()
        cfg.model.patch_backend = patch_backend
        cfg.model.E = 4
        cfg.model.d = 2
        cfg.model.k_in = 2
        cfg.model.pos_strategy = "warmup"
        cfg.model.global_expert_enabled = False
        cfg.train.lr_scheduler_type = "constant"
        cfg.train.enable_usage_runtime_tracking = False
        cfg.train.enable_usage_report = False
        cfg.approx_init.enabled = True
        cfg.approx_init.subset_mode = "warmup_sliding_only"
        cfg.approx_init.steps_per_proj = 2
        cfg.approx_init.batch_size = 2
        cfg.approx_init.identity_chunk_size = 2
        cfg.approx_init.early_stop_patience = 2
        cfg.approx_init.expert_warmup_scaling_enabled = True
        return _finalize_train_config(
            cfg,
            {"lr_scheduler_type", "enable_usage_runtime_tracking", "enable_usage_report"},
        )

    def _fake_model(self, n_layers: int = 2, hidden_size: int = 8, intermediate_size: int = 16):
        return FakeModel(n_layers=n_layers, hidden_size=hidden_size, intermediate_size=intermediate_size)

    def _block_init_cfg(self, *, patch_backend: str = "motn", mode: str = "base_stats_normal"):
        cfg = make_default_config()
        cfg.model.patch_backend = patch_backend
        cfg.model.E = 32
        cfg.model.d = 2
        cfg.model.k_in = 4
        cfg.model.pos_strategy = "random"
        cfg.model.init_gamma = 0.6
        cfg.model.block_init_mode = mode
        cfg.model.block_init_std_scale = 1.0
        cfg.model.block_init_trunc_std = 1.25
        cfg.model.global_expert_enabled = False
        cfg.train.lr_scheduler_type = "constant"
        cfg.train.enable_usage_runtime_tracking = False
        cfg.train.enable_usage_report = False
        cfg.approx_init.enabled = False
        cfg.approx_init.subset_mode = ""
        return _finalize_train_config(
            cfg,
            {"lr_scheduler_type", "enable_usage_runtime_tracking", "enable_usage_report"},
        )

    def _fake_model_with_weight_stats(self, *, n_layers: int = 1, hidden_size: int = 128, intermediate_size: int = 256):
        torch.manual_seed(0)
        model = self._fake_model(n_layers=n_layers, hidden_size=hidden_size, intermediate_size=intermediate_size)
        with torch.no_grad():
            model.model.layers[0].mlp.gate_proj.weight.normal_(mean=-0.35, std=0.07)
            model.model.layers[0].mlp.up_proj.weight.normal_(mean=0.20, std=0.11)
            model.model.layers[0].mlp.down_proj.weight.normal_(mean=0.55, std=0.05)
        return model

    def _dense_stats(self, weight: torch.Tensor):
        w = weight.detach().float()
        return {
            "mean": float(w.mean().item()),
            "std": float(w.std(unbiased=False).item()),
        }

    def _proj_u_values(self, patched_layer, proj_name: str) -> torch.Tensor:
        proj = getattr(patched_layer, proj_name)
        return torch.cat([block.U.detach().float().reshape(-1).cpu() for block in proj.core.blocks], dim=0)

    def test_config_default_and_explicit_backend_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(with_portable_paths({"train": {"lr_scheduler_type": "constant"}})), encoding="utf-8")
            cfg = load_config_from_json(path)
        self.assertEqual(cfg.model.patch_backend, "motn")
        self.assertEqual(cfg.model.block_init_mode, "gamma_normal")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(
                json.dumps(with_portable_paths({"model": {"patch_backend": "adtn_fixed"}, "train": {"lr_scheduler_type": "constant"}})),
                encoding="utf-8",
            )
            cfg = load_config_from_json(path)
        self.assertEqual(cfg.model.patch_backend, "adtn_fixed")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(
                json.dumps(
                    with_portable_paths(
                        {
                            "model": {
                                "patch_backend": "adtn_fixed",
                                "block_init_mode": "base_stats_trunc_normal",
                                "block_init_std_scale": 0.75,
                                "block_init_trunc_std": 1.5,
                            },
                            "train": {"lr_scheduler_type": "constant"},
                        }
                    )
                ),
                encoding="utf-8",
            )
            cfg = load_config_from_json(path)
        patch_cfg = build_patch_model_config(cfg)
        self.assertEqual(cfg.model.block_init_mode, "base_stats_trunc_normal")
        self.assertEqual(patch_cfg["block_init_mode"], "base_stats_trunc_normal")
        self.assertEqual(patch_cfg["block_init_std_scale"], 0.75)
        self.assertEqual(patch_cfg["block_init_trunc_std"], 1.5)

    def test_base_stats_normal_initializes_motn_projection_blocks(self):
        torch.manual_seed(0)
        model = self._fake_model_with_weight_stats()
        dense = {
            name: self._dense_stats(getattr(model.model.layers[0].mlp, name).weight)
            for name in ("gate_proj", "up_proj", "down_proj")
        }
        cfg = build_patch_model_config(self._block_init_cfg(patch_backend="motn", mode="base_stats_normal"))
        torch.manual_seed(0)
        patch_qwen_ffn_layers(model, [0], cfg, device=torch.device("cpu"), dtype=torch.float32)
        layer = model.model.layers[0].mlp

        for name, stats in dense.items():
            with self.subTest(proj=name):
                values = self._proj_u_values(layer, name)
                self.assertAlmostEqual(float(values.mean().item()), stats["mean"], delta=0.02)
                self.assertAlmostEqual(float(values.std(unbiased=False).item()), stats["std"], delta=0.02)

    def test_base_stats_normal_initializes_adtn_fixed_projection_blocks(self):
        torch.manual_seed(0)
        model = self._fake_model_with_weight_stats()
        dense = {
            name: self._dense_stats(getattr(model.model.layers[0].mlp, name).weight)
            for name in ("gate_proj", "up_proj", "down_proj")
        }
        cfg = build_patch_model_config(self._block_init_cfg(patch_backend="adtn_fixed", mode="base_stats_normal"))
        torch.manual_seed(0)
        patch_qwen_ffn_layers(model, [0], cfg, device=torch.device("cpu"), dtype=torch.float32)
        layer = model.model.layers[0].mlp

        for name, stats in dense.items():
            with self.subTest(proj=name):
                values = self._proj_u_values(layer, name)
                self.assertAlmostEqual(float(values.mean().item()), stats["mean"], delta=0.02)
                self.assertAlmostEqual(float(values.std(unbiased=False).item()), stats["std"], delta=0.02)

    def test_base_stats_trunc_normal_clamps_to_projection_bounds(self):
        torch.manual_seed(0)
        model = self._fake_model_with_weight_stats()
        dense = {
            name: self._dense_stats(getattr(model.model.layers[0].mlp, name).weight)
            for name in ("gate_proj", "up_proj", "down_proj")
        }
        fit_cfg = self._block_init_cfg(patch_backend="motn", mode="base_stats_trunc_normal")
        fit_cfg.model.block_init_std_scale = 0.6
        fit_cfg.model.block_init_trunc_std = 1.1
        cfg = build_patch_model_config(fit_cfg)
        torch.manual_seed(0)
        patch_qwen_ffn_layers(model, [0], cfg, device=torch.device("cpu"), dtype=torch.float32)
        layer = model.model.layers[0].mlp

        for name, stats in dense.items():
            effective_std = stats["std"] * fit_cfg.model.block_init_std_scale
            lower = stats["mean"] - fit_cfg.model.block_init_trunc_std * effective_std
            upper = stats["mean"] + fit_cfg.model.block_init_trunc_std * effective_std
            values = self._proj_u_values(layer, name)
            self.assertGreaterEqual(float(values.min().item()), lower - 1e-6)
            self.assertLessEqual(float(values.max().item()), upper + 1e-6)

    def test_global_block_base_stats_scales_std_not_mean(self):
        torch.manual_seed(0)
        model = self._fake_model_with_weight_stats()
        stats = self._dense_stats(model.model.layers[0].mlp.gate_proj.weight)
        fit_cfg = self._block_init_cfg(patch_backend="motn", mode="base_stats_trunc_normal")
        fit_cfg.model.global_expert_enabled = True
        fit_cfg.model.global_expert_init_scale = 0.25
        fit_cfg.model.block_init_std_scale = 0.8
        fit_cfg.model.block_init_trunc_std = 1.0
        cfg = build_patch_model_config(fit_cfg)
        torch.manual_seed(0)
        patch_qwen_ffn_layers(model, [0], cfg, device=torch.device("cpu"), dtype=torch.float32)
        global_u = model.model.layers[0].mlp.gate_proj.core.global_block.U.detach().float()
        effective_std = stats["std"] * fit_cfg.model.block_init_std_scale * fit_cfg.model.global_expert_init_scale
        lower = stats["mean"] - fit_cfg.model.block_init_trunc_std * effective_std
        upper = stats["mean"] + fit_cfg.model.block_init_trunc_std * effective_std
        self.assertGreaterEqual(float(global_u.min().item()), lower - 1e-6)
        self.assertLessEqual(float(global_u.max().item()), upper + 1e-6)
        self.assertLess(abs(float(global_u.mean().item()) - stats["mean"]), 0.03)

    def test_invalid_base_stats_fall_back_to_gamma_normal(self):
        model = self._fake_model(n_layers=1, hidden_size=128, intermediate_size=256)
        with torch.no_grad():
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(model.model.layers[0].mlp, name).weight.fill_(0.5)
        cfg = build_patch_model_config(self._block_init_cfg(patch_backend="motn", mode="base_stats_normal"))
        torch.manual_seed(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            patch_qwen_ffn_layers(model, [0], cfg, device=torch.device("cpu"), dtype=torch.float32)
        self.assertTrue(any("Falling back to gamma_normal" in str(item.message) for item in caught))
        values = self._proj_u_values(model.model.layers[0].mlp, "gate_proj")
        init_std = 1.0 / ((cfg["d"] ** cfg["k_in"]) ** float(cfg["init_gamma"]))
        self.assertAlmostEqual(float(values.mean().item()), 0.0, delta=0.05)
        self.assertAlmostEqual(float(values.std(unbiased=False).item()), init_std, delta=0.05)

    def test_gamma_normal_is_dense_stat_independent(self):
        def build_patched_with_fill(fill_value: float):
            model = self._fake_model(n_layers=1, hidden_size=128, intermediate_size=256)
            with torch.no_grad():
                model.model.layers[0].mlp.gate_proj.weight.fill_(fill_value)
                model.model.layers[0].mlp.up_proj.weight.fill_(fill_value + 1.0)
                model.model.layers[0].mlp.down_proj.weight.fill_(fill_value + 2.0)
            fit_cfg = self._block_init_cfg(patch_backend="motn", mode="gamma_normal")
            cfg = build_patch_model_config(fit_cfg)
            torch.manual_seed(123)
            patch_qwen_ffn_layers(model, [0], cfg, device=torch.device("cpu"), dtype=torch.float32)
            return cfg, self._proj_u_values(model.model.layers[0].mlp, "gate_proj")

        cfg, values_a = build_patched_with_fill(-10.0)
        _, values_b = build_patched_with_fill(10.0)
        init_std = 1.0 / ((cfg["d"] ** cfg["k_in"]) ** float(cfg["init_gamma"]))
        self.assertTrue(torch.equal(values_a, values_b))
        self.assertAlmostEqual(float(values_a.std(unbiased=False).item()), init_std, delta=0.05)

    def test_gamma_normal_does_not_collect_dense_stats(self):
        def fail_if_called(_weight):
            raise AssertionError("dense stats should not be collected for gamma_normal")

        original_motn_helper = motn_model_module.dense_weight_init_stats
        original_fixed_helper = adtn_fixed_module.dense_weight_init_stats
        motn_model_module.dense_weight_init_stats = fail_if_called
        adtn_fixed_module.dense_weight_init_stats = fail_if_called
        try:
            motn_cfg = build_patch_model_config(self._block_init_cfg(patch_backend="motn", mode="gamma_normal"))
            patch_qwen_ffn_layers(
                self._fake_model(n_layers=1, hidden_size=128, intermediate_size=256),
                [0],
                motn_cfg,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            fixed_cfg = build_patch_model_config(self._block_init_cfg(patch_backend="adtn_fixed", mode="gamma_normal"))
            patch_qwen_ffn_layers(
                self._fake_model(n_layers=1, hidden_size=128, intermediate_size=256),
                [0],
                fixed_cfg,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        finally:
            motn_model_module.dense_weight_init_stats = original_motn_helper
            adtn_fixed_module.dense_weight_init_stats = original_fixed_helper

    def test_adtn_fixed_patch_replaces_only_target_mlp(self):
        model = self._fake_model(n_layers=3)
        old_other = model.model.layers[0].mlp
        cfg = build_patch_model_config(self._cfg(patch_backend="adtn_fixed"))
        patched = patch_qwen_ffn_layers(model, [1], cfg, device=torch.device("cpu"), dtype=torch.float32)
        self.assertIs(model.model.layers[0].mlp, old_other)
        self.assertIsInstance(patched.model.layers[1].mlp, ADTNBaselineFFNLayer)
        self.assertIsInstance(patched.model.layers[2].mlp, FakeQwenMLP)

    def test_adtn_fixed_forward_padding_and_shape(self):
        layer = ADTNBaselineFFNLayer(
            FakeQwenMLP(hidden_size=10, intermediate_size=14),
            build_patch_model_config(self._cfg(patch_backend="adtn_fixed")),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        x = torch.randn(2, 3, 10)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 3, 10))
        self.assertEqual(layer.gate_proj.in_pad, 16)
        self.assertEqual(layer.gate_proj.out_pad, 16)

    def test_adtn_fixed_warmup_uses_scaled_fixed_probs(self):
        layer = ADTNBaselineFFNLayer(
            FakeQwenMLP(hidden_size=8, intermediate_size=12),
            build_patch_model_config(self._cfg(patch_backend="adtn_fixed")),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        layer.train()
        num_blocks = int(layer.gate_proj.core.num_blocks)
        slide_indices = list(range(min(2, num_blocks)))
        random_indices = list(range(len(slide_indices), num_blocks))
        warmup_state = {
            "enabled": True,
            "gate_proj": {"slide_indices": slide_indices, "random_indices": random_indices, "random_scale": 0.25},
            "up_proj": {"slide_indices": slide_indices, "random_indices": random_indices, "random_scale": 0.25},
            "down_proj": {"slide_indices": slide_indices, "random_indices": random_indices, "random_scale": 0.25},
        }
        layer.fitmotn_expert_warmup_state = warmup_state

        captured = {}
        original_forward = layer.gate_proj.core.forward

        def wrapped_forward(self, x, *args, **kwargs):
            captured["probs"] = kwargs.get("probs")
            return original_forward(x, *args, **kwargs)

        layer.gate_proj.core.forward = types.MethodType(wrapped_forward, layer.gate_proj.core)
        _ = layer(torch.randn(2, 3, 8))
        probs = captured["probs"]
        self.assertIsNotNone(probs)
        self.assertEqual(tuple(probs.shape), (6, num_blocks))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(6), atol=1e-6))
        self.assertTrue(torch.allclose(probs[0], probs[1], atol=1e-6))
        if random_indices:
            self.assertGreater(float(probs[0, slide_indices[0]]), float(probs[0, random_indices[0]]))

    def test_adtn_fixed_trainable_params_exclude_dense_linears_and_gate(self):
        model = self._fake_model(n_layers=2)
        cfg = build_patch_model_config(self._cfg(patch_backend="adtn_fixed"))
        patch_qwen_ffn_layers(model, [0], cfg, device=torch.device("cpu"), dtype=torch.float32)
        set_trainable_patch_only(model)
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
        self.assertTrue(trainable_names)
        self.assertFalse(any(name.endswith("gate_proj.weight") for name in trainable_names))
        self.assertFalse(any(".core.gate." in name for name in trainable_names))
        layer = model.model.layers[0].mlp
        self.assertFalse(hasattr(layer.gate_proj.core, "gate"))
        self.assertFalse(hasattr(layer.up_proj.core, "gate"))
        self.assertFalse(hasattr(layer.down_proj.core, "gate"))

    def test_adtn_fixed_approx_init_smoke(self):
        model = self._fake_model(n_layers=1, hidden_size=6, intermediate_size=10)
        dense_targets = {
            0: {
                "gate_proj": model.model.layers[0].mlp.gate_proj.weight.detach().cpu().float().clone(),
                "up_proj": model.model.layers[0].mlp.up_proj.weight.detach().cpu().float().clone(),
                "down_proj": model.model.layers[0].mlp.down_proj.weight.detach().cpu().float().clone(),
            }
        }
        cfg = self._cfg(patch_backend="adtn_fixed")
        patch_cfg = build_patch_model_config(cfg)
        patch_qwen_ffn_layers(model, [0], patch_cfg, device=torch.device("cpu"), dtype=torch.float32)
        result = fit_patched_ffn_layer(model.model.layers[0].mlp, dense_targets[0], cfg.approx_init, logger=None)
        self.assertIn("gate_proj", result["projections"])
        self.assertTrue(result["success"] or result["failed"])

    def test_restore_supports_legacy_motn_and_adtn_fixed_metadata(self):
        original_loader = restore_module.load_causal_lm_and_tokenizer

        def fake_loader(*args, **kwargs):
            return self._fake_model(n_layers=1), object(), torch.float32

        restore_module.load_causal_lm_and_tokenizer = fake_loader
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_dir = Path(tmpdir) / "legacy"
                ckpt_dir.mkdir()
                motn_cfg = build_patch_model_config(self._cfg(patch_backend="motn"))
                legacy_model = self._fake_model(n_layers=1)
                patch_qwen_ffn_layers(legacy_model, [0], motn_cfg, device=torch.device("cpu"), dtype=torch.float32)
                legacy_metadata = {
                    "base_model_path": "fake-base",
                    "layers_to_patch": [0],
                    "motn_cfg": motn_cfg,
                    "patch_state_dict": legacy_model.state_dict(),
                }
                save_fitmotn_metadata(ckpt_dir, legacy_metadata)
                model, _, metadata = restore_fitmotn_model(ckpt_dir, device="cpu")
                self.assertEqual(metadata.get("patch_backend", "motn"), "motn")
                self.assertEqual(type(model.model.layers[0].mlp).__name__, "MOTNFFNLayer")

            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_dir = Path(tmpdir) / "adtn"
                ckpt_dir.mkdir()
                patch_cfg = build_patch_model_config(self._cfg(patch_backend="adtn_fixed"))
                adtn_model = self._fake_model(n_layers=1)
                patch_qwen_ffn_layers(adtn_model, [0], patch_cfg, device=torch.device("cpu"), dtype=torch.float32)
                adtn_metadata = {
                    "base_model_path": "fake-base",
                    "layers_to_patch": [0],
                    "patch_backend": "adtn_fixed",
                    "patch_cfg": patch_cfg,
                    "motn_cfg": patch_cfg,
                    "patch_state_dict": adtn_model.state_dict(),
                }
                save_fitmotn_metadata(ckpt_dir, adtn_metadata)
                model, _, metadata = restore_fitmotn_model(ckpt_dir, device="cpu")
                self.assertEqual(metadata["patch_backend"], "adtn_fixed")
                self.assertIsInstance(model.model.layers[0].mlp, ADTNBaselineFFNLayer)
        finally:
            restore_module.load_causal_lm_and_tokenizer = original_loader

    def test_checkpoint_and_run_summary_include_patch_fields(self):
        cfg = self._cfg(patch_backend="adtn_fixed")
        runtime = {
            "run_name": "demo",
            "fit_cfg": cfg,
            "checkpoint_format": "patch_state_only_v2",
            "scheduler_step": 1,
            "patch_backend": "adtn_fixed",
            "patch_cfg": {"patch_backend": "adtn_fixed", "E": 4, "d": 2, "k_in": 2},
            "layers_to_patch": [0],
            "patched_layer_count": 1,
            "resolved_model_dtype": "float32",
            "amp_enabled": False,
            "trainable_params": 10,
            "total_params": 100,
            "trainable_ratio": 0.1,
            "current_stage": None,
            "warmup_updates": 0,
            "lr_scheduler_type": "constant",
            "lr_warmup": False,
            "lr_warmup_steps": 0,
            "resolved_lr_warmup_steps": 0,
            "lr_decay_steps": None,
            "resolved_lr_decay_steps": None,
            "actual_training_steps": 1,
            "scheduler_total_steps": 1,
            "baseline_small_summary": None,
            "baseline_final_summary": None,
            "latest_mid_eval_summary": None,
            "final_full_summary": None,
            "compare_vs_baseline": None,
            "approx_init_summary": None,
            "is_resume_training": False,
            "resume_fitmotn_from": None,
            "resume_stage": "auto",
            "stage2_only_on_resume": True,
            "extra_updates": None,
            "actual_total_updates": 1,
            "stage_a_updates": 1,
            "stage_b_updates": 0,
            "approx_init_skipped_due_to_resume": False,
            "loaded_fitmotn_state_path": None,
            "loaded_checkpoint_metadata": None,
            "resume_load_summary": None,
            "current_model_structure_summary": None,
            "expert_warmup_scaling_summary": None,
            "env_snapshot": {},
            "start_time": 0.0,
            "tokens_per_sec_sum": 0.0,
            "tokens_per_sec_count": 0,
            "max_cuda_mem_peak_alloc_mb": None,
            "recent_train_losses": deque(),
            "mid_eval_records": [],
            "stage_plan": None,
            "current_lr": None,
            "current_param_group_lrs": None,
            "current_temperature": None,
            "gate_trainable": None,
        }
        metadata = build_checkpoint_metadata(runtime, layers_to_patch=[0], motn_cfg=runtime["patch_cfg"], patch_cfg=runtime["patch_cfg"], fit_cfg=cfg)
        self.assertEqual(metadata["patch_backend"], "adtn_fixed")
        self.assertEqual(metadata["patch_cfg"]["patch_backend"], "adtn_fixed")
        summary = build_run_summary(runtime, {"baseline_small": None, "baseline_final": None, "final_full": None, "compare_vs_baseline": None}, final_model_dir="/tmp/final")
        self.assertEqual(summary["patch_backend"], "adtn_fixed")
        self.assertEqual(summary["patch_cfg"]["patch_backend"], "adtn_fixed")


if __name__ == "__main__":
    unittest.main()
