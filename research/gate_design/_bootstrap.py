"""Allow research scripts to run directly from an uninstalled checkout."""
import os
from pathlib import Path
import sys
import types

ROOT = Path(os.environ.get("MOTE_REPO", Path(__file__).resolve().parents[2]))
if "fitmotn" not in sys.modules:
    if not (ROOT / "quantum_gate.py").is_file():
        raise RuntimeError("Run from research/gate_design, or set MOTE_REPO to the checkout")
    package = types.ModuleType("fitmotn")
    package.__path__ = [str(ROOT)]
    package.__package__ = "fitmotn"
    sys.modules["fitmotn"] = package
