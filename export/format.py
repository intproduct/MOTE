from __future__ import annotations

import json
from pathlib import Path

EXPORT_FORMAT_NAME = "fitmotn_hf_export"
EXPORT_FORMAT_VERSION = 1
EXPORT_STAGE_METADATA_ONLY = "metadata_only"
EXPORT_STAGE_HF_ROUNDTRIP = "hf_roundtrip"
EXPORT_MANIFEST_FILENAME = "fitmotn_export_manifest.json"
EXPORT_CONFIG_FILENAME = "fitmotn_export_config.json"
RAW_CHECKPOINT_MARKERS = ("fitmotn_state.pt", "fitmotn_state.json")
FITMOTN_AUTO_MAP = {
    "AutoConfig": "configuration_fitmotn.FitMoTNConfig",
    "AutoModel": "modeling_fitmotn.FitMoTNModel",
    "AutoModelForCausalLM": "modeling_fitmotn.FitMoTNForCausalLM",
}


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _read_json_if_possible(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def is_raw_fitmotn_checkpoint(path: str | Path) -> bool:
    target = _as_path(path)
    return target.is_dir() and any((target / marker).exists() for marker in RAW_CHECKPOINT_MARKERS)


def is_exported_fitmotn_dir(path: str | Path) -> bool:
    target = _as_path(path)
    if not target.is_dir():
        return False
    manifest = _read_json_if_possible(target / EXPORT_MANIFEST_FILENAME)
    config = _read_json_if_possible(target / EXPORT_CONFIG_FILENAME)
    if manifest is None or config is None:
        return False
    return (
        manifest.get("format_name") == EXPORT_FORMAT_NAME
        and manifest.get("format_version") == EXPORT_FORMAT_VERSION
        and "base_model_name_or_path" in config
    )


def classify_model_path(path: str | Path) -> str:
    target = _as_path(path)
    if not target.exists():
        return "missing"
    if not target.is_dir():
        return "unknown"
    if is_exported_fitmotn_dir(target):
        return "exported_fitmotn_dir"
    if is_raw_fitmotn_checkpoint(target):
        return "raw_fitmotn_checkpoint"
    if (target / "config.json").exists():
        return "hf_model_dir"
    return "unknown"


def require_exported_fitmotn_dir(path: str | Path) -> None:
    if is_exported_fitmotn_dir(path):
        return
    kind = classify_model_path(path)
    raise ValueError(
        f"Expected an exported FitMoTN directory at {Path(path)}, got {kind!r}. "
        "For a raw checkpoint, run: "
        "python -m fitmotn.cli.export_hf --checkpoint_dir <raw_checkpoint_dir> --output_dir <export_dir>"
    )
