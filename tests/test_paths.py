from __future__ import annotations

import logging
import os
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

from MOTE.utils.paths import assert_no_unsafe_paths, reset_path_warning_cache, resolve_path


class _Runtime:
    dev_mode = False


class _Cfg:
    runtime = _Runtime()


class PathUtilsTests(unittest.TestCase):
    def setUp(self):
        reset_path_warning_cache()
        self._old_env = {key: os.environ.get(key) for key in ["MODEL_ROOT", "CACHE_ROOT", "PROJECT_ROOT", "FITMOTN_PATH_DEBUG"]}
        os.environ.pop("FITMOTN_PATH_DEBUG", None)

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_path_warning_cache()

    def test_resolve_path_expands_supported_env_vars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MODEL_ROOT"] = str(Path(tmpdir) / "models")
            resolved = resolve_path("${MODEL_ROOT}/Qwen3-8B", key="model.model_path", source="config", cfg=_Cfg(), allow_none=False)

        self.assertEqual(resolved, str((Path(tmpdir) / "models" / "Qwen3-8B").resolve()))

    def test_resolve_path_uses_project_root_for_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PROJECT_ROOT"] = str(Path(tmpdir) / "project")
            resolved = resolve_path("runs/demo", key="output.root_dir", source="config", cfg=_Cfg(), allow_none=False)

        self.assertEqual(resolved, str((Path(tmpdir) / "project" / "runs" / "demo").resolve()))

    def test_prod_missing_env_raises(self):
        os.environ.pop("CACHE_ROOT", None)

        with self.assertRaisesRegex(OSError, "CACHE_ROOT"):
            resolve_path("${CACHE_ROOT}/gsm8k", key="data.gsm8k_cache_path", source="config", cfg=_Cfg(), allow_none=False)

    def test_dev_mode_missing_env_falls_back(self):
        os.environ.pop("CACHE_ROOT", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PROJECT_ROOT"] = str(Path(tmpdir) / "project")
            cfg = _Cfg()
            cfg.runtime = types.SimpleNamespace(dev_mode=True)
            with self.assertLogs("MOTE.utils.paths", level="WARNING") as logs:
                resolved = resolve_path("${CACHE_ROOT}/gsm8k", key="data.gsm8k_cache_path", source="config", cfg=cfg, allow_none=False)

        self.assertEqual(resolved, str((Path(tmpdir) / "project" / "cache" / "gsm8k").resolve()))
        self.assertIn("PATH_RESOLVE_FALLBACK", "\n".join(logs.output))
        self.assertIn({"key": "data.gsm8k_cache_path", "resolved_path": resolved, "source": "config", "severity": "fallback"}, cfg._path_audit_events)

    def test_debug_mode_logs_full_resolve(self):
        os.environ["FITMOTN_PATH_DEBUG"] = "1"
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "model"
            with self.assertLogs("MOTE.utils.paths", level="INFO") as logs:
                resolved = resolve_path(str(target), key="model.model_path", source="config", cfg=_Cfg(), allow_none=False)

        self.assertEqual(resolved, str(target.resolve()))
        self.assertIn("PATH_RESOLVE: model.model_path", "\n".join(logs.output))

    def test_duplicate_work_warning_prints_once(self):
        path = "/" + "work" + "/project/model"
        with self.assertLogs("MOTE.utils.paths", level="WARNING") as logs:
            resolve_path(path, key="model.model_path", source="config", cfg=_Cfg(), allow_none=False)
            resolve_path(path, key="model.model_path", source="config", cfg=_Cfg(), allow_none=False)

        warning_lines = [line for line in logs.output if "PATH_RESOLVE_WARNING" in line]
        self.assertEqual(len(warning_lines), 1)

    def test_home_path_raises(self):
        forbidden = "/" + "home" + "/user/cache"

        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            resolve_path(forbidden, key="data.cache_path", source="config", cfg=_Cfg(), allow_none=False)

    def test_global_safety_check_raises_on_home_path(self):
        cfg = {"data": {"cache_path": "/" + "home" + "/user/cache"}}

        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            assert_no_unsafe_paths(cfg, context="test")


if __name__ == "__main__":
    unittest.main()
