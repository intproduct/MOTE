from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .format import (
    EXPORT_CONFIG_FILENAME,
    EXPORT_FORMAT_NAME,
    EXPORT_FORMAT_VERSION,
    EXPORT_STAGE_HF_ROUNDTRIP,
    EXPORT_MANIFEST_FILENAME,
    EXPORT_STAGE_METADATA_ONLY,
    FITMOTN_AUTO_MAP,
)


HF_WEIGHT_PATTERNS = (
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "model.safetensors",
    "model-*.safetensors",
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


def _has_hf_weight_file(target: Path) -> bool:
    return any(list(target.glob(pattern)) for pattern in HF_WEIGHT_PATTERNS)


def validate_export_layout(path: str | Path) -> ExportValidationResult:
    target = Path(path).expanduser()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    manifest = _load_json(target / EXPORT_MANIFEST_FILENAME, errors)
    export_config = _load_json(target / EXPORT_CONFIG_FILENAME, errors)

    if manifest is not None:
        common_checks = [
            ("format_name", EXPORT_FORMAT_NAME),
            ("format_version", EXPORT_FORMAT_VERSION),
        ]
        for key, expected in common_checks:
            if manifest.get(key) != expected:
                errors.append(_error("invalid_manifest_field", f"{key} must be {expected!r}", target / EXPORT_MANIFEST_FILENAME))
        stage = manifest.get("export_stage")
        if stage == EXPORT_STAGE_METADATA_ONLY:
            for key, expected in [("hf_roundtrip_ready", False), ("vllm_ready", False)]:
                if manifest.get(key) != expected:
                    errors.append(_error("invalid_manifest_field", f"{key} must be {expected!r}", target / EXPORT_MANIFEST_FILENAME))
        elif stage == EXPORT_STAGE_HF_ROUNDTRIP:
            for key, expected in [
                ("hf_roundtrip_ready", True),
                ("exported_code_ready", True),
                ("auto_map_ready", True),
            ]:
                if manifest.get(key) != expected:
                    errors.append(_error("invalid_manifest_field", f"{key} must be {expected!r}", target / EXPORT_MANIFEST_FILENAME))
            if manifest.get("vllm_ready") not in (False, True):
                errors.append(_error("invalid_manifest_field", "vllm_ready must be a boolean", target / EXPORT_MANIFEST_FILENAME))
            if manifest.get("vllm_ready") is True:
                for key, expected in [
                    ("vllm_backend", "transformers"),
                    ("vllm_model_impl", "transformers"),
                    ("vllm_requires_trust_remote_code", True),
                ]:
                    if manifest.get(key) != expected:
                        errors.append(_error("invalid_manifest_field", f"{key} must be {expected!r}", target / EXPORT_MANIFEST_FILENAME))
        else:
            errors.append(_error("invalid_manifest_field", f"export_stage must be {EXPORT_STAGE_METADATA_ONLY!r} or {EXPORT_STAGE_HF_ROUNDTRIP!r}", target / EXPORT_MANIFEST_FILENAME))

    if not (target / "README.md").exists():
        errors.append(_error("missing_file", "Missing required file: README.md", target / "README.md"))
    stage = (manifest or {}).get("export_stage")
    if stage == EXPORT_STAGE_METADATA_ONLY and (target / "config.json").exists():
        warnings.append(
            _error(
                "placeholder_hf_config",
                "config.json exists, but hf_roundtrip_ready=false; this is not a final HF-loadable model directory.",
                target / "config.json",
            )
        )
    if stage == EXPORT_STAGE_HF_ROUNDTRIP:
        hf_config = _load_json(target / "config.json", errors)
        for filename in ("configuration_fitmotn.py", "modeling_fitmotn.py"):
            if not (target / filename).exists():
                errors.append(_error("missing_file", f"Missing required file: {filename}", target / filename))
        if hf_config is not None:
            if hf_config.get("model_type") != "fitmotn":
                errors.append(_error("invalid_hf_config", "config.json model_type must be 'fitmotn'", target / "config.json"))
            if hf_config.get("architectures") != ["FitMoTNForCausalLM"]:
                errors.append(
                    _error(
                        "invalid_hf_config",
                        "config.json architectures must be ['FitMoTNForCausalLM']",
                        target / "config.json",
                    )
                )
            auto_map = hf_config.get("auto_map") or {}
            for key, expected in FITMOTN_AUTO_MAP.items():
                if auto_map.get(key) != expected:
                    errors.append(_error("invalid_hf_config", f"config.json auto_map[{key!r}] must be {expected!r}", target / "config.json"))
            if manifest is not None and manifest.get("vllm_ready") is True and "AutoModel" not in auto_map:
                errors.append(_error("invalid_hf_config", "vLLM-ready config.json must include auto_map['AutoModel']", target / "config.json"))
            patch_cfg = hf_config.get("fitmotn_patch_config")
            if not isinstance(patch_cfg, dict):
                errors.append(_error("invalid_hf_config", "config.json fitmotn_patch_config must be a JSON object", target / "config.json"))
        if not _has_hf_weight_file(target):
            errors.append(_error("missing_file", "HF roundtrip export must contain at least one HF weight file", target))
        if export_config is not None:
            for key, expected in [
                ("hf_roundtrip_ready", True),
                ("exported_code_ready", True),
                ("auto_map_ready", True),
            ]:
                if export_config.get(key) != expected:
                    errors.append(_error("invalid_export_config_field", f"{key} must be {expected!r}", target / EXPORT_CONFIG_FILENAME))
            if export_config.get("vllm_ready") not in (False, True):
                errors.append(_error("invalid_export_config_field", "vllm_ready must be a boolean", target / EXPORT_CONFIG_FILENAME))
            if export_config.get("vllm_ready") is True:
                for key, expected in [
                    ("vllm_backend", "transformers"),
                    ("vllm_model_impl", "transformers"),
                    ("vllm_requires_trust_remote_code", True),
                ]:
                    if export_config.get(key) != expected:
                        errors.append(_error("invalid_export_config_field", f"{key} must be {expected!r}", target / EXPORT_CONFIG_FILENAME))
    return ExportValidationResult(ok=not errors, errors=errors, warnings=warnings)
