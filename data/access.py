from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..runtime import normalize_hf_config

try:
    from datasets import load_dataset, load_from_disk
except Exception:
    load_dataset = None
    load_from_disk = None


def _first_example(ds):
    try:
        return next(iter(ds))
    except StopIteration:
        return None


def _dataset_len(ds):
    try:
        return len(ds)
    except Exception:
        return None


def check_code_dataset_accessible(data_cfg, logger=None) -> bool:
    code_path = Path(str(data_cfg.code_cache_path)).expanduser()
    ready_flag = code_path / "_READY"
    if code_path.exists() and ready_flag.exists():
        return True
    try:
        ds = load_dataset(data_cfg.code_hf_name, normalize_hf_config(data_cfg.code_hf_config), split="train", streaming=True)
        try:
            next(iter(ds))
        except StopIteration:
            pass
        return True
    except ImportError:
        if logger is not None:
            logger.warning("datasets package not installed; cannot pre-check code dataset access. Continue assuming access.")
        return True
    except Exception as exc:
        if logger is not None:
            logger.warning("Code dataset access check failed for %s: %r", data_cfg.code_hf_name, exc)
        return False


def inspect_task_dataset(task, logger=None) -> Dict[str, Any]:
    path = Path(str(task.path)).expanduser()
    ready_flag = path / "_READY"
    info: Dict[str, Any] = {
        "ok": False,
        "reason": "",
        "sample": None,
        "source": "remote_probe",
        "resolved_samples": None,
    }
    if path.exists():
        info["source"] = "cached" if ready_flag.exists() else "cache_probe"
        if load_from_disk is None:
            info["ok"] = True
            info["reason"] = "local dataset path exists; datasets package unavailable so sample validation was skipped"
            return info
        try:
            from .caching import resolve_split

            ds = resolve_split(load_from_disk(str(path)), task.split)
            info["sample"] = _first_example(ds)
            info["resolved_samples"] = _dataset_len(ds)
            info["ok"] = True
            info["reason"] = "local cached dataset ready"
            return info
        except Exception as exc:
            if ready_flag.exists():
                if logger is not None:
                    logger.warning("[%s] cached dataset validation failed at %s: %r", task.name, path, exc)
                info["reason"] = f"cached dataset validation failed: {exc!r}"
                return info
            if logger is not None:
                logger.warning("[%s] cached dataset validation failed at %s: %r", task.name, path, exc)
            info["source"] = "remote_probe"

    if load_dataset is None:
        if logger is not None:
            logger.warning("[%s] datasets package not installed; cannot pre-check task dataset access. Continue assuming access.", task.name)
        info["ok"] = True
        info["reason"] = "datasets package unavailable; skipped access pre-check"
        return info

    try:
        ds = load_dataset(task.hf_name or task.path, normalize_hf_config(task.hf_config), split=task.split, streaming=True)
        info["sample"] = _first_example(ds)
        info["ok"] = True
        info["reason"] = "remote dataset probe ok"
        return info
    except Exception as exc:
        if logger is not None:
            logger.warning("[%s] dataset access check failed for %s config=%s split=%s: %r", task.name, task.hf_name or task.path, normalize_hf_config(task.hf_config), task.split, exc)
        info["reason"] = f"dataset access check failed: {exc!r}"
        return info
