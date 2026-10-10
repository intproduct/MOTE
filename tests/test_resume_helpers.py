from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

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
from MOTE.train.controller import (
    _prepare_run_dir,
    _restore_exact_data_sampling_state,
    _training_arguments_from_kwargs,
    _validate_resume_source,
    _validate_resume_structure,
)


class ResumeHelperTests(unittest.TestCase):
    def test_training_arguments_filters_transformers5_removed_options(self):
        args = _training_arguments_from_kwargs({
            "output_dir": tempfile.mkdtemp(),
            "overwrite_output_dir": False,
            "save_safetensors": False,
            "report_to": [],
        })
        self.assertIsNotNone(args)

    def test_non_resume_nonempty_output_fails_closed(self):
        cfg = make_default_config()
        cfg.output.overwrite_output_dir = False
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir()
            (run_dir / "existing.log").write_text("owned", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "non-resume output directory is not empty"):
                _prepare_run_dir(cfg, run_dir)
            cfg.train.resume_weights_from = "/tmp/checkpoint-1"
            _prepare_run_dir(cfg, run_dir)

    def test_missing_resume_path_raises(self):
        missing = Path(tempfile.gettempdir()) / "definitely_missing_fitmotn_checkpoint"
        with self.assertRaisesRegex(FileNotFoundError, "path does not exist"):
            _validate_resume_source(missing)

    def test_missing_fitmotn_state_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "missing fitmotn_state.pt"):
                _validate_resume_source(tmpdir)

    def test_legacy_exact_resume_uses_deterministic_replay_skip(self):
        class Dataset:
            loaded = False

            def load_sampling_state_dict(self, state):
                self.loaded = True

        cfg = make_default_config()
        cfg.train.resume_checkpoint_from = "/tmp/checkpoint-2000"
        cfg.data.dataloader_num_workers = 0
        cfg.data.source_sampling_mode = "legacy_sequential"
        dataset = Dataset()
        runtime = {}
        logger = SimpleNamespace(info=lambda *args, **kwargs: None)
        restored = _restore_exact_data_sampling_state(
            dataset,
            cfg,
            {
                "data_sampling_state": {
                    "initialized": True,
                    "uncheckpointable_tasks": ["legacy_source"],
                }
            },
            runtime,
            logger,
        )
        self.assertFalse(restored)
        self.assertFalse(dataset.loaded)
        self.assertEqual(runtime["data_sampling_resume"], "restored_via_legacy_replay_skip")

    def test_exact_resume_loads_explicit_sampler_state(self):
        class Dataset:
            loaded = None

            def load_sampling_state_dict(self, state):
                self.loaded = state

        cfg = make_default_config()
        cfg.train.resume_checkpoint_from = "/tmp/checkpoint-2000"
        cfg.data.dataloader_num_workers = 0
        cfg.data.source_sampling_mode = "deterministic_auto"
        state = {"initialized": True, "uncheckpointable_tasks": []}
        dataset = Dataset()
        runtime = {}
        logger = SimpleNamespace(info=lambda *args, **kwargs: None)
        restored = _restore_exact_data_sampling_state(
            dataset,
            cfg,
            {"data_sampling_state": state},
            runtime,
            logger,
        )
        self.assertTrue(restored)
        self.assertIs(dataset.loaded, state)
        self.assertEqual(runtime["data_sampling_resume"], "restored")

    def test_explicit_structure_conflict_raises(self):
        cfg = make_default_config()
        cfg.model.k_in = 7
        setattr(cfg, "_explicit_model_keys", {"k_in"})
        metadata = {"motn_cfg": {"k_in": 5, "topk": cfg.model.topk, "E": cfg.model.E}}

        with self.assertRaisesRegex(ValueError, "checkpoint structure conflicts"):
            _validate_resume_structure(cfg, metadata, [0, 1], metadata["motn_cfg"], n_layers=4)


if __name__ == "__main__":
    unittest.main()
