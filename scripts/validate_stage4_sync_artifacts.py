#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MANIFEST_NAME = "fitmotn_vllm_sync_manifest.json"
STATE_NAME = "fitmotn_vllm_sync_state.json"


def _load_object(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read JSON object {path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"JSON value is not an object: {path}")
        return None
    return value


def validate_sync_root(root: Path, *, min_policy_version: int = 0) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    temporary = sorted(path.name for path in root.glob(".*policy-*.tmp-*")) if root.is_dir() else []
    if not root.is_dir():
        errors.append(f"sync root does not exist: {root}")
    if temporary:
        errors.append(f"incomplete transaction directories remain: {temporary}")

    exports: list[dict[str, Any]] = []
    referenced_raw_names: set[str] = set()
    for export_dir in sorted(root.glob("hf-policy-u*")) if root.is_dir() else []:
        manifest = _load_object(export_dir / MANIFEST_NAME, errors)
        if manifest is None:
            continue
        version = int(manifest.get("policy_version", -1))
        fingerprint = str(manifest.get("policy_fingerprint", ""))
        raw_name = str(manifest.get("raw_checkpoint_name", ""))
        if manifest.get("complete") is not True:
            errors.append(f"export manifest is not committed: {export_dir}")
        if str(manifest.get("export_name", "")) != export_dir.name:
            errors.append(f"export manifest name mismatch: {export_dir}")
        if not fingerprint:
            errors.append(f"export manifest has no policy fingerprint: {export_dir}")
        raw_dir = root / raw_name
        if raw_name:
            referenced_raw_names.add(raw_name)
        if not raw_name or not raw_dir.is_dir():
            errors.append(f"paired raw checkpoint is missing for {export_dir}: {raw_dir}")
        exports.append(
            {
                "policy_version": version,
                "policy_fingerprint": fingerprint,
                "export_dir": str(export_dir),
                "raw_checkpoint_dir": str(raw_dir),
                "roundtrip_validated": bool(manifest.get("roundtrip_validated", False)),
            }
        )
    orphan_raws = sorted(
        path.name for path in root.glob("raw-policy-u*") if path.name not in referenced_raw_names
    ) if root.is_dir() else []
    if orphan_raws:
        errors.append(f"orphan raw checkpoint directories remain: {orphan_raws}")

    state = _load_object(root / STATE_NAME, errors) if root.is_dir() else None
    if state is not None:
        state_export = Path(str(state.get("export_dir", "")))
        state_manifest = _load_object(state_export / MANIFEST_NAME, errors) if state_export else None
        if state_manifest is not None:
            for key in ("policy_version", "policy_fingerprint"):
                if state.get(key) != state_manifest.get(key):
                    errors.append(f"sync state {key} does not match active export manifest")
        if int(state.get("policy_version", -1)) < int(min_policy_version):
            errors.append(
                f"active policy version {state.get('policy_version')} is below required {int(min_policy_version)}"
            )
    if not exports:
        errors.append("no committed hf-policy export was found")
    elif not any(item["roundtrip_validated"] for item in exports):
        warnings.append("none of the retained exports records a full HF roundtrip validation")

    return {
        "ok": not errors,
        "root": str(root),
        "errors": errors,
        "warnings": warnings,
        "temporary_artifacts": temporary,
        "orphan_raw_artifacts": orphan_raws,
        "committed_export_count": len(exports),
        "exports": exports,
        "active_state": state,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate transactional Stage 4G export_reload artifacts")
    parser.add_argument("sync_root")
    parser.add_argument("--min-policy-version", type=int, default=0)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    result = validate_sync_root(Path(args.sync_root), min_policy_version=args.min_policy_version)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
