"""Copy compact experiment evidence without altering frozen outputs.

The manifest records exact hashes for exported files. Large weights/arrays are
inventoried with sizes only; transferring them is a separate, explicit operation.
Existing destinations are refused so an evidence release stays immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
STUDIES = (
    "gate_benchmark_round2", "autoregressive_gate", "routing_stability_round1",
    "operator_geometry_round1", "operator_geometry_round2", "operator_geometry_round3",
)
SKIP_DIRS = {"source", "sources", "source_snapshot", "source_snapshots", "__pycache__", "logs"}
OMIT_NAMES = {"ACTIVE_PROCESS.json", "PIPELINE_LOCK.json", "STATUS.json", "AUTOMATION.json",
              "MONITOR_STATUS.json", "LATEST_HEALTH_AUDIT.json", "live_health.json"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export(destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Evidence release already exists: {destination}")
    # Check all source roots before creating a partial release.
    for study in STUDIES:
        if not (ROOT / "outputs" / study).is_dir():
            raise FileNotFoundError(study)
    destination.mkdir(parents=True)
    included, external, counts = [], [], Counter()
    for study in STUDIES:
        base = ROOT / "outputs" / study
        for source in sorted(base.rglob("*")):
            if not source.is_file():
                continue
            rel = source.relative_to(base)
            if any(part in SKIP_DIRS or part.startswith("source_snapshot") for part in rel.parts):
                continue
            size = source.stat().st_size
            original = source.relative_to(ROOT).as_posix()
            compact = (source.suffix in {".json", ".csv", ".jsonl"}
                       and source.name not in OMIT_NAMES and size <= 8 * 1024 ** 2
                       and not source.name.endswith("_QUEUE.json")
                       and not source.name.endswith("_STATUS.json"))
            if compact:
                target = destination / study / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                digest = sha256(source)
                assert sha256(target) == digest
                included.append({"path": target.relative_to(destination).as_posix(),
                                 "original": original, "bytes": size, "sha256": digest})
                counts[study] += 1
            elif source.suffix in {".pt", ".pth", ".safetensors", ".npz", ".npy", ".bin", ".zip"}:
                external.append({"original": original, "bytes": size,
                                 "sha256": None, "hash_status": "not_rehashed_for_this_publication"})
    result = {"schema": "quantum-evidence-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_workspace": str(ROOT), "counts": dict(counts),
              "exported_bytes": sum(row["bytes"] for row in included),
              "files": included,
              "scope": "Compact original CSV/JSON/JSONL, including negative and failed experiments. "
                       "Historical paths/status embedded in evidence are provenance, not launch instructions.",
              "external_artifacts": "EXTERNAL_ARTIFACTS.json",
              "training_inputs_required_for_new_qwen_work": "Download Qwen and prepare a new frozen corpus; old checkpoints are optional diagnostics."}
    (destination / "MANIFEST.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "EXTERNAL_ARTIFACTS.json").write_text(json.dumps({
        "scope": "Local retained binary artifacts; not present after git pull. Transfer selectively if replay is required.",
        "files": external}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.destination.resolve())
    print(json.dumps({"counts": result["counts"], "bytes": result["exported_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
