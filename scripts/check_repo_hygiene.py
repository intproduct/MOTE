from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "ENV",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

WEIGHT_SUFFIXES = {".pt", ".pth", ".safetensors"}
ALLOWLIST: set[str] = set()


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_archive_path(rel_path: str) -> bool:
    return rel_path == "archive" or rel_path.startswith("archive/")


def scan_repo(root: str | Path, *, allow_archive: bool = False) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    findings: dict[str, list[str]] = {
        "macos": [],
        "python_cache": [],
        "legacy_duplicates": [],
        "weights": [],
    }
    nonfatal: dict[str, list[str]] = {
        "archive_legacy_duplicates": [],
    }

    for path in root_path.rglob("*"):
        try:
            rel = _rel(path, root_path)
        except ValueError:
            continue
        parts = set(path.relative_to(root_path).parts)
        if parts & IGNORED_DIRS:
            continue
        if rel in ALLOWLIST:
            continue

        name = path.name
        if path.is_dir():
            if name == "__MACOSX":
                findings["macos"].append(rel)
            if name == "__pycache__":
                findings["python_cache"].append(rel)
            continue
        if name == ".DS_Store":
            findings["macos"].append(rel)
        elif name.endswith(".pyc"):
            findings["python_cache"].append(rel)
        elif "(1)" in name:
            if allow_archive and _is_archive_path(rel):
                nonfatal["archive_legacy_duplicates"].append(rel)
            else:
                findings["legacy_duplicates"].append(rel)
        elif path.suffix in WEIGHT_SUFFIXES:
            findings["weights"].append(rel)

    for group in findings.values():
        group.sort()
    for group in nonfatal.values():
        group.sort()
    fatal_count = sum(len(group) for group in findings.values())
    return {
        "root": str(root_path),
        "ok": fatal_count == 0,
        "fatal_count": fatal_count,
        "findings": findings,
        "nonfatal": nonfatal,
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"root: {report['root']}")
    for category, paths in report["findings"].items():
        if not paths:
            continue
        print(f"{category}:")
        for path in paths:
            print(f"  {path}")
    for category, paths in report["nonfatal"].items():
        if not paths:
            continue
        print(f"{category} (reported, nonfatal):")
        for path in paths:
            print(f"  {path}")
    if report["ok"]:
        print("ok: no fatal hygiene findings")
    else:
        print(f"failed: {report['fatal_count']} fatal hygiene finding(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check FitMoTN repository hygiene")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--allow-archive", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    report = scan_repo(args.root, allow_archive=bool(args.allow_archive))
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        _print_text(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
