from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
# Keep this checkout ahead of sibling directories that may also be named
# ``fitmotn``. Adding ``_ROOT.parent`` here allowed a spawned subprocess to
# import a different sibling checkout before this package shim.
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

__path__ = [str(Path(__file__).resolve().parent), str(_ROOT)]

from .config.defaults import make_default_config

__all__ = ["make_default_config"]
