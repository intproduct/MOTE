from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .format import (
    EXPORT_CONFIG_FILENAME,
    EXPORT_FORMAT_NAME,
    EXPORT_FORMAT_VERSION,
    EXPORT_MANIFEST_FILENAME,
    EXPORT_STAGE_METADATA_ONLY,
)


@dataclass
class ExportValidationResult:
    ok: bool
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)


def _error(code: str, message: str, path: Path | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"code": code, "message": message}
    if path is not None:
        out["path"] = str(path)
    return out


def _load_json(path: Path, errors: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not path.exists():
        errors.append(_error("missing_file", f"Missing required file: {path.name}", path))
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        errors.append(_error("invalid_json", f"{path.name} is not valid JSON: {exc}", path))
        return None
    if not isinstance(data, dict):
        errors.append(_error("invalid_json_type", f"{path.name} must contain a JSON object", path))
        return None
    return data


def validate_export_layout(path: str | Path) -> ExportValidationResult:
    target = Path(path).expanduser()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    manifest = _load_json(target / EXPORT_MANIFEST_FILENAME, errors)
    _load_json(target / EXPORT_CONFIG_FILENAME, errors)

    if manifest is not None:
        checks = [
            ("format_name", EXPORT_FORMAT_NAME),
            ("format_version", EXPORT_FORMAT_VERSION),
            ("export_stage", EXPORT_STAGE_METADATA_ONLY),
            ("hf_roundtrip_ready", False),
            ("vllm_ready", False),
        ]
        for key, expected in checks:
            if manifest.get(key) != expected:
                errors.append(_error("invalid_manifest_field", f"{key} must be {expected!r}", target / EXPORT_MANIFEST_FILENAME))

    if not (target / "README.md").exists():
        errors.append(_error("missing_file", "Missing required file: README.md", target / "README.md"))
    if (target / "config.json").exists() and (manifest or {}).get("hf_roundtrip_ready") is False:
        warnings.append(
            _error(
                "placeholder_hf_config",
                "config.json exists, but hf_roundtrip_ready=false; this is not a final HF-loadable model directory.",
                target / "config.json",
            )
        )
    return ExportValidationResult(ok=not errors, errors=errors, warnings=warnings)
