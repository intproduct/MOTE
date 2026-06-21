from __future__ import annotations

import sys
import tempfile
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
from MOTE.train.controller import _validate_resume_source, _validate_resume_structure


class ResumeHelperTests(unittest.TestCase):
    def test_missing_resume_path_raises(self):
        missing = Path(tempfile.gettempdir()) / "definitely_missing_fitmotn_checkpoint"
        with self.assertRaisesRegex(FileNotFoundError, "path does not exist"):
            _validate_resume_source(missing)

    def test_missing_fitmotn_state_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "missing fitmotn_state.pt"):
                _validate_resume_source(tmpdir)

    def test_explicit_structure_conflict_raises(self):
        cfg = make_default_config()
        cfg.model.k_in = 7
        setattr(cfg, "_explicit_model_keys", {"k_in"})
        metadata = {"motn_cfg": {"k_in": 5, "topk": cfg.model.topk, "E": cfg.model.E}}

        with self.assertRaisesRegex(ValueError, "checkpoint structure conflicts"):
            _validate_resume_structure(cfg, metadata, [0, 1], metadata["motn_cfg"], n_layers=4)


if __name__ == "__main__":
    unittest.main()
