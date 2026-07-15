from __future__ import annotations

import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg
if "MOTE.config" not in sys.modules:
    config_pkg = types.ModuleType("MOTE.config")
    config_pkg.__path__ = [str(ROOT / "config")]
    sys.modules["MOTE.config"] = config_pkg
if "MOTE.data" not in sys.modules:
    data_pkg = types.ModuleType("MOTE.data")
    data_pkg.__path__ = [str(ROOT / "data")]
    sys.modules["MOTE.data"] = data_pkg
if "MOTE.train" not in sys.modules:
    train_pkg = types.ModuleType("MOTE.train")
    train_pkg.__path__ = [str(ROOT / "train")]
    sys.modules["MOTE.train"] = train_pkg

from MOTE.config.defaults import make_default_config
from MOTE.config.loader import _finalize_train_config, load_config_from_json
from MOTE.data.tokenization import make_supervised_example
from MOTE.train.stages import build_stage_plan, resolve_stage_temperature, set_optimizer_stage_lrs
from MOTE.train.stages import MutableStageState
from MOTE.train.callbacks import MOTNScheduleCallback
from MOTE.train.observability import UpdateBatchMeta
from transformers import TrainerControl, TrainerState
from MOTE.train.trainer import compute_loss_observability


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


class TinyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 251 + 1 for ch in str(text)]


class StageControlObservabilityTests(unittest.TestCase):
    def _cfg(self, **train_values):
        cfg = make_default_config()
        for key, value in train_values.items():
            setattr(cfg.train, key, value)
        return _finalize_train_config(cfg, set(train_values.keys()))

    def _optimizer(self):
        router = torch.nn.Parameter(torch.tensor([1.0]))
        blocks = torch.nn.Parameter(torch.tensor([1.0]))
        return torch.optim.SGD(
            [
                {"params": [router], "lr": 1e-5, "name": "router"},
                {"params": [blocks], "lr": 1e-5, "name": "blocks"},
            ]
        )

    def test_config_new_fields_validate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(with_portable_paths({"train": {"lr_scheduler_type": "constant", "stage_a_block_lr": 1e-4}})), encoding="utf-8")
            cfg = load_config_from_json(path)
        self.assertEqual(cfg.train.stage_a_block_lr, 1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(with_portable_paths({"train": {"stage_a_block_lr": 0.0}})), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stage_a_block_lr must be > 0"):
                load_config_from_json(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps({"train": {"lr_scheduler_type": "linear", "stage_b_router_lr": 1e-5}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stage-specific learning rates"):
                load_config_from_json(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps({"train": {"temperature_schedule_type": "bad"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "temperature_schedule_type"):
                load_config_from_json(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps({"train": {"final_answer_weight": 0}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "final_answer_weight"):
                load_config_from_json(path)

    def test_stage_lrs_apply_and_fallback(self):
        cfg = self._cfg(
            lr_scheduler_type="constant",
            block_lr=7e-5,
            router_lr=8e-5,
            stage_a_block_lr=1e-4,
            stage_a_router_lr=2e-4,
            stage_b_block_lr=5e-5,
            stage_b_router_lr=6e-5,
        )
        opt = self._optimizer()
        set_optimizer_stage_lrs(opt, cfg.train, "stage_a_recover")
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 2e-4)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 1e-4)
        set_optimizer_stage_lrs(opt, cfg.train, "stage_b_taskaware")
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 6e-5)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 5e-5)

        fallback = self._cfg(lr_scheduler_type="constant", block_lr=7e-5, router_lr=8e-5)
        set_optimizer_stage_lrs(opt, fallback.train, "stage_a_recover")
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 8e-5)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 7e-5)

    def test_stage_temperature_uses_stage_local_bounds(self):
        cfg = self._cfg(
            stage_a_begin_t=2.0,
            stage_a_end_t=1.0,
            stage_b_begin_t=1.0,
            stage_b_end_t=1.0,
            temperature_schedule_type="cosine",
            steps=10,
            epochs=0,
            stage_a_ratio=0.5,
        )
        plan = build_stage_plan(cfg.train)
        self.assertAlmostEqual(resolve_stage_temperature(cfg.train, plan.stages[0], 0), 2.0)
        self.assertAlmostEqual(resolve_stage_temperature(cfg.train, plan.stages[1], plan.stages[1].start_update), 1.0)

        resume = self._cfg(
            resume_fitmotn_from="/tmp/ckpt",
            stage2_only_on_resume=True,
            extra_updates=10,
            stage_b_begin_t=1.3,
            stage_b_end_t=1.1,
        )
        plan = build_stage_plan(resume.train)
        self.assertTrue(plan.stages[0].name.startswith("stage_b_"))
        self.assertAlmostEqual(resolve_stage_temperature(resume.train, plan.stages[0], 0), 1.3)

        constant = self._cfg(
            temperature_schedule_type="constant",
            stage_b_end_t=0.9,
            steps=10,
            epochs=0,
            stage_a_ratio=0.5,
        )
        plan = build_stage_plan(constant.train)
        self.assertAlmostEqual(resolve_stage_temperature(constant.train, plan.stages[1], plan.stages[1].start_update), 0.9)

    def test_final_answer_weight_marker(self):
        tok = TinyTokenizer()
        ex = make_supervised_example(
            tok,
            "Q",
            "reason #### 42",
            128,
            final_answer_weight_enabled=True,
            final_answer_weight=2.0,
            final_answer_marker="####",
        )
        self.assertIn("loss_weights", ex)
        self.assertGreater(int(ex["loss_weights"].gt(1.0).sum().item()), 0)

        missing = make_supervised_example(
            tok,
            "Q",
            "reason answer",
            128,
            final_answer_weight_enabled=True,
            final_answer_weight=2.0,
            final_answer_marker="####",
        )
        self.assertEqual(int(missing["loss_weights"].gt(1.0).sum().item()), 0)

    def test_weighted_loss_default_matches_standard_and_bucket_metrics(self):
        logits = torch.zeros(2, 4, 7)
        labels = torch.tensor([[1, 2, 3, -100], [1, 4, 5, 6]])
        standard, metrics = compute_loss_observability(logits, labels, buckets=["gsm8k_core", "aux_reasoning"])
        weighted, _ = compute_loss_observability(
            logits,
            labels,
            buckets=["gsm8k_core", "aux_reasoning"],
            loss_weights=(labels != -100).to(torch.float32),
        )
        self.assertAlmostEqual(float(standard.item()), float(weighted.item()), places=7)
        self.assertAlmostEqual(float(standard.item()), math.log(7), places=6)
        self.assertIn("loss/gsm8k_core", metrics)
        self.assertIn("loss/aux_reasoning", metrics)
        self.assertGreater(metrics["tokens/gsm8k_core"], 0)
        self.assertGreater(metrics["tokens/aux_reasoning"], 0)

        weights = (labels != -100).to(torch.float32)
        weights[1, 2:] = 2.0
        _, weighted_metrics = compute_loss_observability(logits, labels, buckets=["gsm8k_core", "aux_reasoning"], loss_weights=weights)
        self.assertGreater(weighted_metrics["final_answer_weighted_tokens"], 0)

    def test_stage_transition_forces_checkpoint_save(self):
        cfg = make_default_config()
        cfg.model.patch_backend = "adtn_fixed"
        cfg.train.epochs = 0
        cfg.train.steps = 10
        cfg.train.stage_a_ratio = 0.5
        cfg.train.save_on_stage_transition = True
        cfg.train.usage_light_every = 0
        cfg.train.usage_light_jsonl_every = 0
        cfg.train.usage_report_every = 0
        cfg.train.eval_every_updates = 0
        plan = build_stage_plan(cfg.train)
        stage_state = MutableStageState(plan)
        model = torch.nn.Linear(2, 2)
        model.fitmotn_runtime = {
            "batch_size": 1,
            "grad_accum": 1,
            "seq_len": 1,
            "tokens_per_update": 1,
            "update_batch_meta": UpdateBatchMeta(),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            callback = MOTNScheduleCallback(
                cfg,
                stage_state,
                root / "train.jsonl",
                root / "train_light.jsonl",
                root / "usage.jsonl",
                root / "eval.jsonl",
            )
            state = TrainerState(global_step=5)
            control = callback.on_step_end(
                None,
                state,
                TrainerControl(),
                model=model,
                optimizer=None,
            )

        self.assertTrue(control.should_save)
        self.assertTrue(model.fitmotn_runtime["checkpoint_stage_boundary_pending"])
        self.assertEqual(model.fitmotn_runtime["checkpoint_stage_boundary_name"], "stage_a_recover_end")


if __name__ == "__main__":
    unittest.main()
