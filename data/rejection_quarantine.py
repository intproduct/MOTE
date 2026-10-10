from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import content_hash
from .release import tokenizer_fingerprint


FORMAT = "fitmotn_dynamic_rejection_quarantine_v1"
MANIFEST_FILENAME = "manifest.json"
REJECTIONS_FILENAME = "rejections.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_policy_payload(tasks: Iterable[Any]) -> list[dict[str, Any]]:
    payload = []
    for task in tasks:
        if not bool(getattr(task, "supports_skip", False)):
            continue
        payload.append(
            {
                "name": str(task.name),
                "kind": str(task.kind),
                "path": str(task.path),
                "split": str(task.split),
                "hf_name": getattr(task, "hf_name", None),
                "hf_config": getattr(task, "hf_config", None),
                "dataset_name": getattr(task, "dataset_name", None),
                "max_samples": getattr(task, "max_samples", None),
                "length_policy": dict(getattr(task, "length_policy", {}) or {}),
                "supervision_mode": getattr(task, "supervision_mode", None),
                "trace_selection_policy": getattr(task, "trace_selection_policy", None),
            }
        )
    return sorted(payload, key=lambda value: value["name"])


def task_policy_fingerprint(tasks: Iterable[Any]) -> str:
    return content_hash(task_policy_payload(tasks))


def validate_rejection_quarantine(root_dir: str | Path) -> dict[str, Any]:
    root = Path(root_dir).expanduser().resolve()
    errors: list[str] = []
    manifest_path = root / MANIFEST_FILENAME
    records_path = root / REJECTIONS_FILENAME
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"missing {MANIFEST_FILENAME}"], "manifest": None}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [f"invalid manifest: {exc}"], "manifest": None}
    if manifest.get("format") != FORMAT:
        errors.append(f"unsupported quarantine format: {manifest.get('format')!r}")
    fingerprint = manifest.get("manifest_fingerprint")
    fingerprint_payload = dict(manifest)
    fingerprint_payload.pop("manifest_fingerprint", None)
    if fingerprint != content_hash(fingerprint_payload):
        errors.append("manifest fingerprint mismatch")
    if not records_path.is_file():
        errors.append(f"missing {REJECTIONS_FILENAME}")
    else:
        file_info = dict(manifest.get("files", {}).get(REJECTIONS_FILENAME) or {})
        if file_info.get("sha256") != sha256_file(records_path):
            errors.append(f"sha256 mismatch: {REJECTIONS_FILENAME}")
        try:
            with records_path.open("r", encoding="utf-8") as handle:
                count = sum(1 for line in handle if line.strip())
        except OSError as exc:
            errors.append(f"cannot read {REJECTIONS_FILENAME}: {exc}")
        else:
            if int(file_info.get("rows", -1)) != count:
                errors.append(f"row count mismatch: {REJECTIONS_FILENAME}")
    return {"ok": not errors, "errors": errors, "manifest": manifest}


class RejectionQuarantine:
    def __init__(
        self,
        root_dir: str | Path,
        tasks: Iterable[Any],
        *,
        tokenizer: Any | None = None,
        max_length: int | None = None,
    ):
        report = validate_rejection_quarantine(root_dir)
        if not report["ok"]:
            raise RuntimeError(f"invalid rejection quarantine: {report['errors']}")
        self.root = Path(root_dir).expanduser().resolve()
        self.manifest = dict(report["manifest"])
        expected_policy = task_policy_fingerprint(tasks)
        if self.manifest.get("task_policy_fingerprint") != expected_policy:
            raise RuntimeError("rejection quarantine task-policy fingerprint mismatch")
        if tokenizer is not None:
            expected_tokenizer = tokenizer_fingerprint(tokenizer)
            if self.manifest.get("tokenizer_fingerprint") != expected_tokenizer:
                raise RuntimeError("rejection quarantine tokenizer fingerprint mismatch")
        if max_length is not None and int(self.manifest.get("max_length", -1)) != int(max_length):
            raise RuntimeError("rejection quarantine max-length mismatch")
        self._allowed: set[tuple[str, str, str]] = set()
        with (self.root / REJECTIONS_FILENAME).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                key = (str(value.get("task")), str(value.get("reason")), str(value.get("raw_content_hash")))
                if not all(key):
                    raise RuntimeError(f"invalid rejection quarantine record at line {line_number}")
                if key in self._allowed:
                    raise RuntimeError(f"duplicate rejection quarantine record at line {line_number}")
                self._allowed.add(key)

    def allows(self, task: str, reason: str, raw_example: Mapping[str, Any]) -> bool:
        return (str(task), str(reason), content_hash(dict(raw_example))) in self._allowed

    def validate_source(self, task: Any, raw: Any) -> None:
        expected = dict(self.manifest.get("sources", {}).get(str(task.name)) or {})
        if not expected:
            raise RuntimeError(f"rejection quarantine is missing source metadata for task {task.name}")
        try:
            observed_rows = len(raw)
        except Exception:
            observed_rows = None
        observed_fingerprint = getattr(raw, "_fingerprint", None)
        if expected.get("rows") is not None and observed_rows != int(expected["rows"]):
            raise RuntimeError(
                f"rejection quarantine source row-count mismatch for {task.name}: "
                f"expected={expected.get('rows')} observed={observed_rows}"
            )
        if expected.get("dataset_fingerprint") and observed_fingerprint != expected.get("dataset_fingerprint"):
            raise RuntimeError(
                f"rejection quarantine source fingerprint mismatch for {task.name}: "
                f"expected={expected.get('dataset_fingerprint')} observed={observed_fingerprint}"
            )
