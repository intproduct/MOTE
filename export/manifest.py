from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .format import EXPORT_FORMAT_NAME, EXPORT_FORMAT_VERSION, EXPORT_STAGE_METADATA_ONLY


def read_raw_checkpoint_json(checkpoint_dir: str | Path) -> tuple[dict[str, Any], list[str]]:
    path = Path(checkpoint_dir).expanduser() / "fitmotn_state.json"
    if not path.exists():
        return {}, [f"fitmotn_state.json not found at {path}; metadata summary is limited"]
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return {}, [f"fitmotn_state.json is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return {}, ["fitmotn_state.json did not contain a JSON object"]
    return data, []


def _pick_base_model(metadata: dict[str, Any], override: str | None) -> str:
    base = override or metadata.get("base_model_path")
    if not base:
        raise ValueError("base_model_name_or_path could not be inferred; pass --base_model for this checkpoint")
    return str(base)


def _pick_tokenizer(metadata: dict[str, Any], base_model: str, override: str | None) -> str:
    return str(override or metadata.get("tokenizer_path") or base_model)


def _fitmotn_state_files(checkpoint_dir: Path) -> list[str]:
    return [name for name in ["fitmotn_state.pt", "fitmotn_state.json"] if (checkpoint_dir / name).exists()]


def build_patch_metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    patch_cfg = metadata.get("patch_cfg") or metadata.get("motn_cfg") or {}
    if not isinstance(patch_cfg, dict):
        patch_cfg = {}
    layers = metadata.get("layers_to_patch") or []
    return {
        "checkpoint_format": metadata.get("checkpoint_format"),
        "checkpoint_name": metadata.get("checkpoint_name"),
        "global_step": metadata.get("global_step"),
        "layers_to_patch": list(layers) if isinstance(layers, list) else layers,
        "patch_backend": metadata.get("patch_backend", patch_cfg.get("patch_backend", "motn")),
        "patch_config": patch_cfg,
        "resolved_model_dtype": metadata.get("resolved_model_dtype"),
        "updates_done": metadata.get("updates_done"),
    }


def build_export_payloads(
    checkpoint_dir: str | Path,
    *,
    base_model: str | None = None,
    tokenizer: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(checkpoint_dir).expanduser()
    metadata, warnings = read_raw_checkpoint_json(checkpoint_path)
    base_model_name_or_path = _pick_base_model(metadata, base_model)
    tokenizer_source = _pick_tokenizer(metadata, base_model_name_or_path, tokenizer)
    patch_summary = build_patch_metadata_summary(metadata)
    fitmotn_patch_config = patch_summary.get("patch_config") or {}

    manifest = {
        "base_model_name_or_path": base_model_name_or_path,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "export_stage": EXPORT_STAGE_METADATA_ONLY,
        "fitmotn_state_files": _fitmotn_state_files(checkpoint_path),
        "format_name": EXPORT_FORMAT_NAME,
        "format_version": EXPORT_FORMAT_VERSION,
        "hf_roundtrip_ready": False,
        "next_steps": [
            "Stage 4B will add configuration_fitmotn.py, modeling_fitmotn.py, auto_map, weights, and HF AutoModel roundtrip support.",
            "Stage 4C will add vLLM offline runner support.",
        ],
        "patch_metadata_summary": patch_summary,
        "source_checkpoint_dir": str(checkpoint_path),
        "tokenizer_source": tokenizer_source,
        "vllm_ready": False,
        "warnings": warnings,
    }
    export_config = {
        "auto_map_ready": False,
        "backend": patch_summary.get("patch_backend") or "motn",
        "base_config_summary": {
            "base_model_name_or_path": base_model_name_or_path,
            "resolved_model_dtype": metadata.get("resolved_model_dtype"),
        },
        "base_model_name_or_path": base_model_name_or_path,
        "exported_code_ready": False,
        "fitmotn_patch_config": fitmotn_patch_config,
        "layers_to_patch": patch_summary.get("layers_to_patch") or [],
        "notes": [
            "Stage 4A export is metadata-only.",
            "This is not a Hugging Face config.json and is not loadable with AutoModelForCausalLM yet.",
        ],
        "tokenizer_source": tokenizer_source,
    }
    return manifest, export_config


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
