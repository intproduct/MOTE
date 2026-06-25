from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import torch as tc

try:
    import numpy as np
except Exception:
    np = None


def to_jsonable(obj: Any):
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tc.dtype):
        return str(obj)
    if isinstance(obj, tc.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if np is not None:
        if isinstance(obj, np.dtype):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    return obj


def build_fitmotn_json_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    json_metadata = dict(metadata)
    json_metadata.pop("state_dict", None)
    json_metadata.pop("patch_state_dict", None)
    return to_jsonable(json_metadata)


def extract_patch_state_dict(model: tc.nn.Module, layer_idxs: Iterable[int]) -> Dict[str, tc.Tensor]:
    prefixes = tuple(f"model.layers.{int(idx)}.mlp." for idx in sorted(set(int(idx) for idx in layer_idxs)))
    state_dict = model.state_dict()
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
        if key.startswith(prefixes)
    }


def get_restore_state_dict(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if "patch_state_dict" in metadata and metadata["patch_state_dict"] is not None:
        return metadata["patch_state_dict"]
    if "state_dict" in metadata and metadata["state_dict"] is not None:
        return metadata["state_dict"]
    raise KeyError("metadata does not contain patch_state_dict or state_dict")


def save_fitmotn_metadata(output_dir: str | Path, metadata: Dict[str, Any]) -> Path:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    state_path = output_path / "fitmotn_state.pt"
    tc.save(metadata, state_path)
    with (output_path / "fitmotn_state.json").open("w", encoding="utf-8") as f:
        json.dump(build_fitmotn_json_metadata(metadata), f, ensure_ascii=False, indent=2)
    return state_path


def load_fitmotn_metadata(ckpt_dir: str | Path) -> Dict[str, Any]:
    path = Path(ckpt_dir).resolve() / "fitmotn_state.pt"
    if not path.exists():
        raise FileNotFoundError(f"fitmotn metadata not found: {path}")
    try:
        metadata = tc.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        metadata = tc.load(path, map_location="cpu")
    if isinstance(metadata, dict) and "checkpoint_format" not in metadata:
        metadata["checkpoint_format"] = "legacy_full_state"
    return metadata
