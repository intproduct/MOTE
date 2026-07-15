from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from MOTE.checkpointing import write_checkpoint_manifest
from MOTE.config.defaults import make_default_config
from MOTE.diagnostics.config_doctor import inspect_config


class ConfigDoctorTests(unittest.TestCase):
    def test_reports_generated_and_retained_checkpoint_counts(self):
        cfg = make_default_config()
        cfg.train.epochs = 0
        cfg.train.steps = 10
        cfg.train.save_every_updates = 3
        cfg.train.stage_a_ratio = 0.5
        cfg.train.save_on_stage_transition = True
        cfg.train.checkpoint_keep_last_n = 2
        cfg.train.checkpoint_keep_every_n = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output.root_dir = str(Path(tmpdir) / "runs")
            cfg.model.model_path = str(Path(tmpdir) / "missing-model")
            result = inspect_config(cfg)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["sft"]["periodic_checkpoint_count"], 3)
        self.assertEqual(result["sft"]["stage_boundary_checkpoint_count"], 1)
        self.assertEqual(result["sft"]["generated_checkpoint_count_before_retention"], 4)
        self.assertEqual(result["sft"]["estimated_retained_checkpoint_count_including_final"], 4)
        self.assertEqual(result["sft"]["estimated_peak_checkpoint_count_during_atomic_save"], 5)

    def test_invalid_exact_resume_is_an_error(self):
        cfg = make_default_config()
        cfg.train.epochs = 0
        cfg.train.steps = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint-1"
            checkpoint.mkdir()
            write_checkpoint_manifest(checkpoint, checkpoint_kind="sft", update_step=1)
            cfg.train.resume_checkpoint_from = str(checkpoint)
            cfg.output.root_dir = str(root / "runs")
            cfg.model.model_path = str(root / "missing-model")
            result = inspect_config(cfg)

        self.assertFalse(result["ok"])
        self.assertTrue(any("exact-resume checkpoint is invalid" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
