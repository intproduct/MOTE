from __future__ import annotations

import json
import hashlib
import os
import shutil
import time
import uuid
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import torch as tc

try:
    import numpy as np
except Exception:
    np = None


CHECKPOINT_MANIFEST_NAME = "fitmotn_checkpoint_manifest.json"
CHECKPOINT_MANIFEST_FORMAT = "fitmotn_checkpoint_manifest_v1"
RL_TRAINING_STATE_NAME = "rl_training_state.pt"


def to_jsonable(obj: Any):
    if is_dataclass(obj):
        out = {}
        for field in fields(obj):
            if hasattr(obj, field.name):
                out[field.name] = to_jsonable(getattr(obj, field.name))
        return out
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


def _sha256_file(path: Path, *, hash_max_bytes: int) -> Optional[str]:
    size = int(path.stat().st_size)
    if int(hash_max_bytes) >= 0 and size > int(hash_max_bytes):
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_file_inventory(
    checkpoint_dir: str | Path,
    *,
    hash_max_bytes: int = 64 * 1024 * 1024,
) -> list[Dict[str, Any]]:
    root = Path(checkpoint_dir).resolve()
    inventory = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == CHECKPOINT_MANIFEST_NAME:
            continue
        stat = path.stat()
        inventory.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": int(stat.st_size),
                "sha256": _sha256_file(path, hash_max_bytes=int(hash_max_bytes)),
            }
        )
    return inventory


def infer_checkpoint_capabilities(checkpoint_dir: str | Path) -> Dict[str, bool]:
    root = Path(checkpoint_dir)
    has_fitmotn = (root / "fitmotn_state.pt").is_file()
    has_hf_state = (root / "trainer_state.json").is_file()
    has_optimizer = any((root / name).is_file() for name in ("optimizer.pt", "optimizer.bin"))
    has_scheduler = (root / "scheduler.pt").is_file()
    has_rng = (root / "rng_state.pth").is_file() or any(root.glob("rng_state_*.pth"))
    has_rl_state = (root / RL_TRAINING_STATE_NAME).is_file()
    return {
        "weights_resume": bool(has_fitmotn or (root / "config.json").is_file()),
        "sft_exact_resume": bool(has_fitmotn and has_hf_state and has_optimizer and has_scheduler and has_rng),
        "rl_exact_resume": bool(has_fitmotn and has_rl_state),
        "has_fitmotn_state": bool(has_fitmotn),
        "has_optimizer_state": bool(has_optimizer or has_rl_state),
        "has_scheduler_state": bool(has_scheduler),
        "has_trainer_state": bool(has_hf_state or has_rl_state),
        "has_rng_state": bool(has_rng or has_rl_state),
    }


def write_checkpoint_manifest(
    checkpoint_dir: str | Path,
    *,
    checkpoint_kind: str,
    update_step: int,
    stage_name: Optional[str] = None,
    stage_boundary: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
    hash_max_bytes: int = 64 * 1024 * 1024,
) -> Path:
    root = Path(checkpoint_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    manifest = {
        "format": CHECKPOINT_MANIFEST_FORMAT,
        "complete": True,
        "checkpoint_kind": str(checkpoint_kind),
        "update_step": int(update_step),
        "stage_name": None if stage_name is None else str(stage_name),
        "stage_boundary": bool(stage_boundary),
        "created_at_unix": time.time(),
        "capabilities": infer_checkpoint_capabilities(root),
        "files": checkpoint_file_inventory(root, hash_max_bytes=int(hash_max_bytes)),
        "extra": to_jsonable(dict(extra or {})),
    }
    target = root / CHECKPOINT_MANIFEST_NAME
    temporary = root / f".{CHECKPOINT_MANIFEST_NAME}.tmp-{os.getpid()}"
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_checkpoint_manifest(checkpoint_dir: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(checkpoint_dir).resolve() / CHECKPOINT_MANIFEST_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def validate_checkpoint(
    checkpoint_dir: str | Path,
    *,
    verify_hashes: bool = True,
    require_exact_resume: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(checkpoint_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    manifest = load_checkpoint_manifest(root)
    if not root.is_dir():
        errors.append(f"checkpoint directory does not exist: {root}")
    if manifest is None:
        errors.append(f"missing or invalid {CHECKPOINT_MANIFEST_NAME}")
        capabilities = infer_checkpoint_capabilities(root) if root.is_dir() else {}
    else:
        if manifest.get("format") != CHECKPOINT_MANIFEST_FORMAT:
            errors.append(f"unsupported checkpoint manifest format: {manifest.get('format')!r}")
        if manifest.get("complete") is not True:
            errors.append("checkpoint manifest is not committed")
        raw_capabilities = manifest.get("capabilities")
        if not isinstance(raw_capabilities, dict):
            errors.append("checkpoint manifest capabilities must be an object")
            capabilities = {}
        else:
            capabilities = dict(raw_capabilities)
        files = manifest.get("files")
        if not isinstance(files, list):
            errors.append("checkpoint manifest files must be a list")
            files = []
        for item in files:
            if not isinstance(item, dict):
                errors.append("checkpoint manifest contains an invalid file entry")
                continue
            relative = item.get("path")
            relative_path = Path(str(relative))
            path = (root / relative_path).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                errors.append(f"unsafe checkpoint file path: {relative}")
                continue
            if not path.is_file():
                errors.append(f"missing checkpoint file: {relative}")
                continue
            actual_size = int(path.stat().st_size)
            if actual_size != int(item.get("size_bytes", -1)):
                errors.append(
                    f"checkpoint file size mismatch: {relative}: "
                    f"manifest={item.get('size_bytes')} actual={actual_size}"
                )
            expected_hash = item.get("sha256")
            if verify_hashes and expected_hash:
                actual_hash = _sha256_file(path, hash_max_bytes=-1)
                if actual_hash != expected_hash:
                    errors.append(f"checkpoint file hash mismatch: {relative}")
    if root.is_dir() and not (root / "fitmotn_state.pt").is_file():
        warnings.append("fitmotn_state.pt is absent; FitMoTN weight resume is unavailable")
    if require_exact_resume:
        key = f"{str(require_exact_resume).strip().lower()}_exact_resume"
        if not bool(capabilities.get(key, False)):
            errors.append(f"checkpoint does not support requested {require_exact_resume} exact resume")
        if str(require_exact_resume).strip().lower() == "rl" and (root / RL_TRAINING_STATE_NAME).is_file():
            try:
                try:
                    rl_state = tc.load(root / RL_TRAINING_STATE_NAME, map_location="cpu", weights_only=False)
                except TypeError:
                    rl_state = tc.load(root / RL_TRAINING_STATE_NAME, map_location="cpu")
                required = {
                    "optimizer_state_dict",
                    "update_step",
                    "micro_step",
                    "optimizer_micro_step",
                    "data_pos",
                    "python_random_state",
                    "torch_rng_state",
                    "reference_source",
                }
                if not isinstance(rl_state, dict) or rl_state.get("format") != "fitmotn_rl_training_state_v1":
                    errors.append("unsupported RL exact-resume state format")
                else:
                    missing = sorted(required - set(rl_state))
                    if missing:
                        errors.append(f"incomplete RL exact-resume state: missing={missing}")
                    sampler_state = rl_state.get("sampler_state")
                    if sampler_state is not None:
                        sampler_required = {
                            "shuffle",
                            "seed",
                            "record_count",
                            "dataset_fingerprint",
                            "epoch",
                            "position",
                            "samples_seen",
                            "order",
                            "rng_state",
                        }
                        if not isinstance(sampler_state, dict) or sampler_state.get("format") != "fitmotn_rl_sampler_state_v1":
                            errors.append("unsupported RL sampler state format")
                        else:
                            sampler_missing = sorted(sampler_required - set(sampler_state))
                            if sampler_missing:
                                errors.append(f"incomplete RL sampler state: missing={sampler_missing}")
            except Exception as exc:
                errors.append(f"failed to read RL exact-resume state: {exc}")
    return {
        "ok": not errors,
        "checkpoint_dir": str(root),
        "errors": errors,
        "warnings": warnings,
        "manifest": manifest,
        "capabilities": capabilities,
    }


def cleanup_checkpoint_transactions(parent: str | Path, *, max_age_sec: float = 3600.0) -> int:
    root = Path(parent).resolve()
    if not root.is_dir():
        return 0
    removed = 0
    now = time.time()
    for backup in list(root.glob(".checkpoint-*.backup-*")):
        try:
            target_name = backup.name[len(".checkpoint-"):].split(".backup-", 1)[0]
            target = root / target_name
            if target.exists():
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
            else:
                backup.replace(target)
            removed += 1
        except OSError:
            continue
    for path in list(root.glob(".checkpoint-*.tmp-*")) + list(root.glob(".checkpoint-*.delete-*")):
        try:
            age = max(0.0, now - path.stat().st_mtime)
            if float(max_age_sec) > 0.0 and age < float(max_age_sec):
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def commit_prepared_checkpoint(
    prepared_dir: str | Path,
    target_dir: str | Path,
    *,
    checkpoint_kind: str,
    update_step: int,
    stage_name: Optional[str] = None,
    stage_boundary: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    prepared = Path(prepared_dir).resolve()
    target = Path(target_dir).resolve()
    if prepared.parent != target.parent:
        raise ValueError("prepared and target checkpoint directories must share a parent for atomic rename")
    write_checkpoint_manifest(
        prepared,
        checkpoint_kind=checkpoint_kind,
        update_step=int(update_step),
        stage_name=stage_name,
        stage_boundary=bool(stage_boundary),
        extra=extra,
    )
    report = validate_checkpoint(prepared, verify_hashes=True)
    if not report["ok"]:
        raise RuntimeError(f"prepared checkpoint validation failed: {report['errors']}")
    backup = target.parent / f".checkpoint-{target.name}.backup-{uuid.uuid4().hex}"
    if target.exists():
        target.replace(backup)
    try:
        prepared.replace(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.replace(target)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    return target


def transactional_save_checkpoint(
    target_dir: str | Path,
    writer: Callable[[Path], None],
    *,
    checkpoint_kind: str,
    update_step: int,
    stage_name: Optional[str] = None,
    stage_boundary: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
    temp_max_age_sec: float = 3600.0,
) -> Path:
    target = Path(target_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    cleanup_checkpoint_transactions(target.parent, max_age_sec=float(temp_max_age_sec))
    prepared = target.parent / f".checkpoint-{target.name}.tmp-{uuid.uuid4().hex}"
    prepared.mkdir(parents=False, exist_ok=False)
    try:
        writer(prepared)
        return commit_prepared_checkpoint(
            prepared,
            target,
            checkpoint_kind=checkpoint_kind,
            update_step=int(update_step),
            stage_name=stage_name,
            stage_boundary=bool(stage_boundary),
            extra=extra,
        )
    except Exception:
        shutil.rmtree(prepared, ignore_errors=True)
        raise


def transactional_update_checkpoint(
    target_dir: str | Path,
    updater: Callable[[Path], None],
    *,
    checkpoint_kind: str,
    update_step: int,
    stage_name: Optional[str] = None,
    stage_boundary: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
    temp_max_age_sec: float = 3600.0,
) -> Path:
    target = Path(target_dir).expanduser().resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {target}")

    def writer(prepared: Path) -> None:
        shutil.copytree(target, prepared, dirs_exist_ok=True)
        updater(prepared)

    return transactional_save_checkpoint(
        target,
        writer,
        checkpoint_kind=checkpoint_kind,
        update_step=int(update_step),
        stage_name=stage_name,
        stage_boundary=bool(stage_boundary),
        extra=extra,
        temp_max_age_sec=float(temp_max_age_sec),
    )


def _checkpoint_step(path: Path) -> int:
    manifest = load_checkpoint_manifest(path) or {}
    if manifest.get("update_step") is not None:
        return int(manifest["update_step"])
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def prune_checkpoints(
    checkpoint_root: str | Path,
    *,
    keep_last_n: int,
    keep_every_n: int = 0,
    preserve_stage_boundaries: bool = True,
) -> Dict[str, Any]:
    root = Path(checkpoint_root).resolve()
    checkpoints = sorted(
        [path for path in root.glob("checkpoint-*") if path.is_dir()],
        key=_checkpoint_step,
    ) if root.is_dir() else []
    keep: set[Path] = set(checkpoints[-max(0, int(keep_last_n)):]) if int(keep_last_n) > 0 else set()
    for path in checkpoints:
        step = _checkpoint_step(path)
        manifest = load_checkpoint_manifest(path) or {}
        if int(keep_every_n) > 0 and step > 0 and step % int(keep_every_n) == 0:
            keep.add(path)
        if preserve_stage_boundaries and bool(manifest.get("stage_boundary", False)):
            keep.add(path)
    removed = []
    for path in checkpoints:
        if path in keep:
            continue
        tombstone = root / f".checkpoint-{path.name}.delete-{uuid.uuid4().hex}"
        path.replace(tombstone)
        shutil.rmtree(tombstone)
        removed.append(str(path))
    return {
        "kept": [str(path) for path in checkpoints if path in keep],
        "removed": removed,
    }
