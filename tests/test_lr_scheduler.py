from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

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

from MOTE.config.defaults import make_default_config
from MOTE.config.loader import _finalize_train_config
from MOTE.train import runtime as runtime_module
from MOTE.train.runtime import build_scheduler_builder


class LRSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_get_constant_schedule = runtime_module.get_constant_schedule
        cls._orig_get_constant_schedule_with_warmup = runtime_module.get_constant_schedule_with_warmup
        cls._orig_get_linear_schedule_with_warmup = runtime_module.get_linear_schedule_with_warmup
        cls._orig_get_cosine_schedule_with_warmup = runtime_module.get_cosine_schedule_with_warmup
        runtime_module.get_constant_schedule = cls._constant_schedule
        runtime_module.get_constant_schedule_with_warmup = cls._constant_with_warmup_schedule
        runtime_module.get_linear_schedule_with_warmup = cls._linear_with_warmup_schedule
        runtime_module.get_cosine_schedule_with_warmup = cls._cosine_with_warmup_schedule

    @classmethod
    def tearDownClass(cls):
        runtime_module.get_constant_schedule = cls._orig_get_constant_schedule
        runtime_module.get_constant_schedule_with_warmup = cls._orig_get_constant_schedule_with_warmup
        runtime_module.get_linear_schedule_with_warmup = cls._orig_get_linear_schedule_with_warmup
        runtime_module.get_cosine_schedule_with_warmup = cls._orig_get_cosine_schedule_with_warmup

    @staticmethod
    def _constant_schedule(optimizer):
        return LambdaLR(optimizer, lambda _: 1.0)

    @staticmethod
    def _constant_with_warmup_schedule(optimizer, num_warmup_steps: int):
        def lr_lambda(current_step: int):
            if num_warmup_steps <= 0:
                return 1.0
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _linear_with_warmup_schedule(optimizer, num_warmup_steps: int, num_training_steps: int):
        def lr_lambda(current_step: int):
            if num_warmup_steps > 0 and current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            if current_step >= num_training_steps:
                return 0.0
            return max(
                0.0,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
            )

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _cosine_with_warmup_schedule(optimizer, num_warmup_steps: int, num_training_steps: int):
        def lr_lambda(current_step: int):
            if num_warmup_steps > 0 and current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            if current_step >= num_training_steps:
                return 0.0
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item()))

        return LambdaLR(optimizer, lr_lambda)

    def _make_cfg(
        self,
        *,
        scheduler_type: str = "linear",
        lr_warmup: bool = False,
        lr_warmup_steps: int = 0,
        lr_decay_steps: int | None = None,
        block_lr: float = 1e-5,
        router_lr: float = 2e-5,
    ):
        cfg = make_default_config()
        cfg.train.lr_scheduler_type = scheduler_type
        cfg.train.lr_warmup = lr_warmup
        cfg.train.lr_warmup_steps = lr_warmup_steps
        cfg.train.lr_decay_steps = lr_decay_steps
        cfg.train.block_lr = block_lr
        cfg.train.router_lr = router_lr
        return _finalize_train_config(cfg, {"lr_scheduler_type", "lr_warmup", "lr_warmup_steps", "lr_decay_steps", "block_lr", "router_lr"})

    def _make_optimizer(self, block_lr: float = 1e-5, router_lr: float = 2e-5):
        router = torch.nn.Parameter(torch.tensor([1.0]))
        blocks = torch.nn.Parameter(torch.tensor([1.0]))
        return torch.optim.SGD(
            [
                {"params": [router], "lr": router_lr, "name": "router"},
                {"params": [blocks], "lr": block_lr, "name": "blocks"},
            ]
        )

    @staticmethod
    def _group_lrs(optimizer):
        return {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups}

    def _step_scheduler(self, optimizer, scheduler, steps: int):
        history = []
        for _ in range(steps):
            optimizer.step()
            scheduler.step()
            history.append(self._group_lrs(optimizer))
        return history

    def test_linear_default_matches_actual_training_steps(self):
        cfg = self._make_cfg(scheduler_type="linear", lr_warmup=False, lr_decay_steps=None)
        optimizer = self._make_optimizer()
        builder, metadata = build_scheduler_builder(cfg, total_updates=20)
        scheduler = builder(optimizer, num_training_steps=20)
        self.assertEqual(metadata["scheduler_total_steps"], 20)
        self.assertEqual(metadata["resolved_lr_warmup_steps"], 0)

        history = self._step_scheduler(optimizer, scheduler, 20)
        self.assertAlmostEqual(history[-1]["blocks"], 0.0, places=12)
        self.assertAlmostEqual(history[-1]["router"], 0.0, places=12)

    def test_linear_with_large_decay_steps_stays_near_initial_lr(self):
        cfg = self._make_cfg(
            scheduler_type="linear",
            lr_warmup=True,
            lr_warmup_steps=500,
            lr_decay_steps=999999,
        )
        optimizer = self._make_optimizer()
        builder, _ = build_scheduler_builder(cfg, total_updates=22000)
        scheduler = builder(optimizer, num_training_steps=22000)

        history = self._step_scheduler(optimizer, scheduler, 22000)
        self.assertAlmostEqual(history[499]["blocks"], 1e-5, places=12)
        self.assertGreater(history[-1]["blocks"], 0.97e-5)
        self.assertGreater(history[-1]["router"], 1.94e-5)

    def test_constant_scheduler_keeps_group_lrs_constant(self):
        cfg = self._make_cfg(scheduler_type="constant", lr_warmup=False)
        optimizer = self._make_optimizer()
        builder, metadata = build_scheduler_builder(cfg, total_updates=50)
        scheduler = builder(optimizer, num_training_steps=50)

        self.assertEqual(metadata["resolved_lr_warmup_steps"], 0)
        history = self._step_scheduler(optimizer, scheduler, 50)
        for item in history:
            self.assertAlmostEqual(item["blocks"], 1e-5, places=12)
            self.assertAlmostEqual(item["router"], 2e-5, places=12)

    def test_scheduler_preserves_param_group_ratio(self):
        cfg = self._make_cfg(
            scheduler_type="cosine",
            lr_warmup=True,
            lr_warmup_steps=3,
            lr_decay_steps=15,
            block_lr=1e-5,
            router_lr=2e-5,
        )
        optimizer = self._make_optimizer(block_lr=1e-5, router_lr=2e-5)
        builder, _ = build_scheduler_builder(cfg, total_updates=15)
        scheduler = builder(optimizer, num_training_steps=15)

        history = self._step_scheduler(optimizer, scheduler, 15)
        for idx in [0, 2, 7, 13]:
            ratio = history[idx]["router"] / history[idx]["blocks"]
            self.assertAlmostEqual(ratio, 2.0, places=10)

    def test_invalid_scheduler_type_raises_clear_error(self):
        cfg = make_default_config()
        cfg.train.lr_scheduler_type = "invalid"
        with self.assertRaisesRegex(ValueError, "train.lr_scheduler_type must be one of"):
            _finalize_train_config(cfg, {"lr_scheduler_type"})


if __name__ == "__main__":
    unittest.main()
