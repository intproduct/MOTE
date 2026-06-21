from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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
from MOTE.train import controller
from MOTE.train import rl_controller


class RLWrapperTests(unittest.TestCase):
    def _cfg(self):
        cfg = make_default_config()
        cfg.rl.enabled = True
        cfg.rl.run_after_sft = True
        cfg.rl.max_steps = 1
        cfg.train.lr_scheduler_type = "constant"
        return cfg

    def test_run_after_sft_uses_final_model_dir_without_mutating_cfg(self):
        cfg = self._cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            final_model = Path(tmpdir) / "run" / "final_model"
            final_model.mkdir(parents=True)
            sft_result = {"run_dir": str(final_model.parent), "final_model_dir": str(final_model), "run_summary": "summary.json"}
            captured = {}

            def fake_rl(rl_cfg):
                captured["resume_from"] = rl_cfg.rl.resume_from
                captured["same_object"] = rl_cfg is cfg
                return {"final_model_dir": str(Path(tmpdir) / "rl" / "final_model")}

            with patch.object(controller, "run_fitmotn_training", return_value=sft_result):
                with patch.object(rl_controller, "run_fitmotn_rl_training", side_effect=fake_rl):
                    result = rl_controller.run_fitmotn_training_and_optional_rl(cfg)

        self.assertEqual(result["final_model_dir"], sft_result["final_model_dir"])
        self.assertIn("rl_result", result)
        self.assertEqual(captured["resume_from"], str(Path(sft_result["final_model_dir"]).resolve()))
        self.assertFalse(captured["same_object"])
        self.assertIsNone(cfg.rl.resume_from)

    def test_run_after_sft_can_infer_final_model_from_run_dir(self):
        cfg = self._cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            final_model = run_dir / "final_model"
            final_model.mkdir(parents=True)
            captured = {}

            def fake_rl(rl_cfg):
                captured["resume_from"] = rl_cfg.rl.resume_from
                return {"rl_dir": str(Path(tmpdir) / "rl")}

            with patch.object(controller, "run_fitmotn_training", return_value={"run_dir": str(run_dir)}):
                with patch.object(rl_controller, "run_fitmotn_rl_training", side_effect=fake_rl):
                    result = rl_controller.run_fitmotn_training_and_optional_rl(cfg)

        self.assertEqual(captured["resume_from"], str(final_model.resolve()))
        self.assertEqual(result["run_dir"], str(run_dir))
        self.assertIn("rl_result", result)


if __name__ == "__main__":
    unittest.main()
