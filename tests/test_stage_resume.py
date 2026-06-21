from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

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
from MOTE.train.stages import build_stage_plan


class StageResumeTests(unittest.TestCase):
    def test_default_config_keeps_stage_a(self):
        cfg = _finalize_train_config(make_default_config(), set())

        plan = build_stage_plan(cfg.train)

        self.assertGreater(plan.total_updates, 0)
        self.assertGreater(plan.stage_a_updates, 0)
        self.assertEqual(plan.stage_a_updates + plan.stage_b_updates, plan.total_updates)
        self.assertEqual(plan.stages[0].name, "stage_a_recover")

    def test_resume_stage2_only_uses_extra_updates(self):
        cfg = make_default_config()
        cfg.train.resume_fitmotn_from = "/tmp/existing-final-model"
        cfg.train.stage2_only_on_resume = True
        cfg.train.resume_stage = "auto"
        cfg.train.extra_updates = 10
        cfg = _finalize_train_config(cfg, {"resume_fitmotn_from", "stage2_only_on_resume", "resume_stage", "extra_updates"})

        plan = build_stage_plan(cfg.train)

        self.assertEqual(plan.total_updates, 10)
        self.assertEqual(plan.stage_a_updates, 0)
        self.assertEqual(plan.stage_b_updates, 10)
        self.assertEqual(len(plan.stages), 1)
        self.assertTrue(plan.stages[0].name.startswith("stage_b_"))
        self.assertEqual(plan.stages[0].start_update, 0)
        self.assertEqual(plan.stages[0].end_update, 10)


if __name__ == "__main__":
    unittest.main()
