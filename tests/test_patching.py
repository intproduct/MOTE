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

from MOTE.patching import resolve_layer_idxs


class ResolveLayerIdxsTests(unittest.TestCase):
    def test_fixed_and_flexible_modes(self):
        self.assertEqual(resolve_layer_idxs(32, "last_quarter"), list(range(24, 32)))
        self.assertEqual(resolve_layer_idxs(32, "range:4-11"), list(range(4, 12)))
        self.assertEqual(resolve_layer_idxs(32, "range:12-19"), list(range(12, 20)))
        self.assertEqual(resolve_layer_idxs(32, "layers:0,3,7,15"), [0, 3, 7, 15])

    def test_normalizes_case_and_whitespace(self):
        self.assertEqual(resolve_layer_idxs(32, " Range:4-11 "), list(range(4, 12)))
        self.assertEqual(resolve_layer_idxs(32, " layers: 0, 3, 7, 15 "), [0, 3, 7, 15])

    def test_invalid_range_modes_raise(self):
        for mode in ["range:16-7", "range:0-32", "range:-1-5"]:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "layers_to_patch"):
                    resolve_layer_idxs(32, mode)

    def test_invalid_layers_modes_raise(self):
        for mode in ["layers:", "layers:1,abc,3", "layers:1,32", "layers:1,1"]:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "layers_to_patch"):
                    resolve_layer_idxs(32, mode)


if __name__ == "__main__":
    unittest.main()
