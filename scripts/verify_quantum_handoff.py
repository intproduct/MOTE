"""Verify the published evidence hashes and research-plan consistency using stdlib."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_evidence(base):
    base = Path(base).resolve()
    manifest = json.loads((base / "MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    for row in manifest["files"]:
        path = (base / row["path"]).resolve()
        if not path.is_relative_to(base):
            errors.append(f"Unsafe evidence path: {row['path']}")
        elif not path.is_file():
            errors.append(f"Missing: {row['path']}")
        elif path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            errors.append(f"Hash/size mismatch: {row['path']}")
    return {"checked_files": len(manifest["files"]), "errors": errors}


def verify_plans(base):
    base = Path(base)
    common = json.loads((base / "COMMON_PROTOCOL.json").read_text(encoding="utf-8"))
    errors = []
    if common["is_training_cli_config"] or common["architecture"]["gate_freeze_steps"] != 0:
        errors.append("The design must be distinct from CLI configs and jointly train from step one")
    if common["matrix"]["planned_train_runs_per_formal_model"] != 15:
        errors.append("Expected four routed methods plus dense control, each with three seeds")
    for name in ["pro4000", "v100_pair", "a800", "bw1000"]:
        path = base / "plans" / f"{name}.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan["machine"] != name or plan["training_cli_ready"]:
            errors.append(f"Incorrect machine/status: {name}")
        if plan["output_prefix"] != f"outputs/qwen_joint/{name}":
            errors.append(f"Output collision risk: {name}")
        if not (base / "agents" / f"{name}.md").is_file():
            errors.append(f"Missing agent instruction: {name}")
    return {"profiles": 4, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    evidence = verify_evidence(args.root / "research/experiment_evidence_20261030")
    snapshot = verify_evidence(args.root / "research/operator_geometry_runtime_snapshot_20261030")
    plans = verify_plans(args.root / "research/qwen_mote_joint_handoff_20261030")
    errors = evidence["errors"] + snapshot["errors"] + plans["errors"]
    print(json.dumps({"passed": not errors, "evidence": evidence, "frozen_sources": snapshot,
                      "plans": plans}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
