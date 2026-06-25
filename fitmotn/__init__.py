from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

__path__ = [str(Path(__file__).resolve().parent), str(_ROOT)]

from .config.defaults import make_default_config

__all__ = ["make_default_config"]
