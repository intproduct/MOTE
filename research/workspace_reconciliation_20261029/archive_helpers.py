"""Preserve experiment helper source that otherwise lives in ignored outputs.

Copies source only, never runs experiments or changes their original artifacts.
"""

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GROUPS = (
    "gate_benchmark", "gate_benchmark_round2", "autoregressive_gate",
    "routing_stability", "routing_stability_round1",
)


def main():
    entries = []
    for group in GROUPS:
        base = ROOT / "outputs" / group
        for source in sorted(base.rglob("*")):
            if source.suffix not in {".py", ".ps1"} or not source.is_file():
                continue
            # Skip routine frozen copies and generated test/cache artifacts.
            # Named historical snapshots remain part of the provenance archive.
            if any(part in {"source_snapshot", "__pycache__", "data"} or part.startswith("pytest") for part in source.relative_to(base).parts):
                continue
            data = source.read_bytes()
            if source.suffix == ".py":
                ast.parse(data.decode("utf-8-sig"), filename=str(source))
            destination = HERE / "archived_helpers" / group / source.relative_to(base)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != data:
                raise RuntimeError(f"Refusing to overwrite a differing archive: {destination}")
            destination.write_bytes(data)
            entries.append({
                "source": source.relative_to(ROOT).as_posix(),
                "archive": destination.relative_to(ROOT).as_posix(),
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            })

    frozen = []
    for group in ("autoregressive_gate", "routing_stability_round1"):
        for manifest in sorted((ROOT / "outputs" / group / "formal").glob("*/frozen_sources.json")):
            hashes = json.loads(manifest.read_text(encoding="utf-8-sig"))
            checks = [
                {"path": path, "expected": digest,
                 "actual": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
                for path, digest in hashes.items()
            ]
            if any(item["expected"] != item["actual"] for item in checks):
                raise RuntimeError(f"Frozen experiment source changed: {manifest}")
            frozen.append({"manifest": manifest.relative_to(ROOT).as_posix(), "all_match": True, "files": checks})

    report = {
        "canonical_workspace": str(ROOT), "branch": "mote-quantum",
        "archive_policy": "Historical helper sources only; original output paths retained. Do not execute archive copies without reviewing original paths and experiment protocol.",
        "helpers_count": len(entries), "helpers": entries,
        "frozen_source_checks": frozen,
    }
    (HERE / "source_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archived_helpers": len(entries), "frozen_manifests": len(frozen), "all_frozen_sources_match": True}))


if __name__ == "__main__":
    main()
