from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg

from MOTE import audit, checkpointing


@dataclass
class LegacyConfigLike:
    model: str = "base"
    train: str = "sft"
    rl: dict = field(default_factory=dict)


class JsonableCompatTests(unittest.TestCase):
    def test_audit_to_jsonable_skips_missing_dataclass_fields(self):
        obj = LegacyConfigLike()
        delattr(obj, "rl")

        payload = audit.to_jsonable(obj)

        self.assertEqual(payload, {"model": "base", "train": "sft"})

    def test_checkpointing_to_jsonable_skips_missing_dataclass_fields(self):
        obj = LegacyConfigLike()
        delattr(obj, "rl")

        payload = checkpointing.to_jsonable({"fit_cfg": obj})

        self.assertEqual(payload, {"fit_cfg": {"model": "base", "train": "sft"}})


if __name__ == "__main__":
    unittest.main()
