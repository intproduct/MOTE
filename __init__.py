from __future__ import annotations

import sys
from pathlib import Path

_FITMOTN_DIR = Path(__file__).resolve().parent
_MOTN_DIR = _FITMOTN_DIR.parent
if str(_MOTN_DIR) not in sys.path:
    sys.path.insert(0, str(_MOTN_DIR))

from .config.defaults import make_default_config

__all__ = ["make_default_config"]
