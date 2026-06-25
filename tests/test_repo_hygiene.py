from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_repo_hygiene.py"


def test_hygiene_checker_json_output_for_clean_tmpdir(tmp_path):
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(tmp_path), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["fatal_count"] == 0
    assert set(report["findings"]) == {"macos", "python_cache", "legacy_duplicates", "weights"}


def test_hygiene_checker_detects_common_bad_files(tmp_path):
    (tmp_path / ".DS_Store").write_text("", encoding="utf-8")
    cache = tmp_path / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "mod.cpython-310.pyc").write_bytes(b"bad")
    (tmp_path / "file (1).py").write_text("print('legacy')\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(CHECKER), str(tmp_path), "--json"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert ".DS_Store" in report["findings"]["macos"]
    assert "pkg/__pycache__" in report["findings"]["python_cache"]
    assert "file (1).py" in report["findings"]["legacy_duplicates"]


def test_pyproject_metadata_and_default_dependencies():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "fitmotn"
    deps = {dep.split("[", 1)[0].split(" ", 1)[0].lower() for dep in data["project"]["dependencies"]}
    assert {"torch", "transformers", "datasets", "numpy", "tqdm"}.issubset(deps)
    assert "vllm" not in deps
    assert "pytest" in data["project"]["optional-dependencies"]["dev"]
    packages = set(data["tool"]["setuptools"]["packages"])
    assert "fitmotn.cli" in packages
    assert "fitmotn.rl" in packages
    assert not any(pkg.startswith("archive") or pkg.startswith("tests") for pkg in packages)


def test_main_cli_modules_import_under_fitmotn_name():
    for module_name in [
        "fitmotn.cli.train",
        "fitmotn.cli.train_rl",
        "fitmotn.cli.eval_hf",
        "fitmotn.cli.build_boundary_gsm8k",
    ]:
        module = importlib.import_module(module_name)
        assert module is not None
