"""Data entry points must work without another test preloading task modules."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("package", ["fitmotn", "MOTE"])
@pytest.mark.parametrize("module", ["data.release", "data.training_release", "data.mixed_iterable"])
def test_data_entry_point_import_in_fresh_process(package, module):
    root = Path(__file__).resolve().parents[1]
    script = (
        "import importlib, runpy; "
        "runpy.run_path('conftest.py'); "
        f"importlib.import_module({(package + '.' + module)!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root,
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
